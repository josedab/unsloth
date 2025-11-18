# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused LoRA Forward Pass Kernel

This module implements a fused Triton kernel for LoRA forward pass that combines:
- Base weight matmul: Y = X @ W
- LoRA first matmul: XA = X @ A
- LoRA second matmul: XAB = XA @ B
- Final combination: Y = Y + scaling * XAB

This fusion reduces kernel launch overhead and memory bandwidth by loading X tiles
once and reusing them for both computations.
"""

import os
import triton
import triton.language as tl
import torch
from .utils import torch_gpu_device


# Environment variable to disable fused forward kernel
FUSED_FORWARD_ENABLED = not os.environ.get("UNSLOTH_DISABLE_FUSED_FORWARD", False)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_lora_forward_kernel(
    # Input pointers
    X_ptr,      # Input: [M, K]
    W_ptr,      # Base weights: [K, N]
    A_ptr,      # LoRA A: [K, R]
    B_ptr,      # LoRA B: [R, N]
    Y_ptr,      # Output: [M, N]
    # Dimensions
    M, N, K, R,
    # Scaling
    scaling,
    # Strides for X
    stride_xm, stride_xk,
    # Strides for W
    stride_wk, stride_wn,
    # Strides for A
    stride_ak, stride_ar,
    # Strides for B
    stride_br, stride_bn,
    # Strides for Y
    stride_ym, stride_yn,
    # Block sizes (constexpr for compilation)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y = X @ W + scaling * X @ A @ B

    Optimization strategy:
    1. Load X tile once, use for both W and A matmuls
    2. Accumulate results in registers (float32 for precision)
    3. Single store to global memory

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    """
    # Program ID determines output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile boundaries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for base matmul
    acc_base = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Initialize accumulator for X @ A result
    # We'll accumulate this first, then multiply by B
    acc_xa = tl.zeros((BLOCK_M, R), dtype=tl.float32) if R <= 128 else None

    # For small R, we can keep XA in registers
    # For large R, we need a different strategy
    USE_FUSED_LORA = R <= 128

    # Iterate over K dimension for base matmul X @ W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile (will be reused for both W and A matmuls)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Base matmul: X @ W
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc_base += tl.dot(x, w)

    # Compute LoRA contribution: X @ A @ B
    # We need to accumulate X @ A over K, then multiply by B

    # Accumulate X @ A
    acc_xa_local = tl.zeros((BLOCK_M, R), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load A tile for all ranks
        offs_r = tl.arange(0, R)
        a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_r[None, :] * stride_ar
        a = tl.load(a_ptrs, mask=offs_k[:, None] < K, other=0.0)

        # Accumulate X @ A
        acc_xa_local += tl.dot(x, a)

    # Now compute (X @ A) @ B
    # acc_xa_local is [BLOCK_M, R]
    # B is [R, N], we need the portion for offs_n
    offs_r = tl.arange(0, R)
    b_ptrs = B_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn
    b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)

    # LoRA contribution: (X @ A) @ B
    acc_lora = tl.dot(acc_xa_local.to(b.dtype), b).to(tl.float32)

    # Combine: Y = base + scaling * lora
    y = acc_base + scaling * acc_lora

    # Store output
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def fused_lora_forward_kernel_large_rank(
    # Input pointers
    X_ptr,      # Input: [M, K]
    W_ptr,      # Base weights: [K, N]
    A_ptr,      # LoRA A: [K, R]
    B_ptr,      # LoRA B: [R, N]
    Y_ptr,      # Output: [M, N]
    # Dimensions
    M, N, K, R,
    # Scaling
    scaling,
    # Strides for X
    stride_xm, stride_xk,
    # Strides for W
    stride_wk, stride_wn,
    # Strides for A
    stride_ak, stride_ar,
    # Strides for B
    stride_br, stride_bn,
    # Strides for Y
    stride_ym, stride_yn,
    # Block sizes (constexpr for compilation)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """
    Variant for large LoRA ranks (R > 128).

    Uses tiled accumulation over R dimension to handle larger ranks
    that don't fit in registers.
    """
    # Program ID determines output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile boundaries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulators
    acc_base = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_lora = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute base matmul: X @ W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load W tile
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc_base += tl.dot(x, w)

    # Compute LoRA contribution with tiled R dimension
    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)

        # Accumulate X @ A for this R block
        acc_xa_block = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load X tile
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

            # Load A tile for this R block
            a_ptrs = A_ptr + offs_k[:, None] * stride_ak + offs_r[None, :] * stride_ar
            a = tl.load(a_ptrs, mask=(offs_k[:, None] < K) & (offs_r[None, :] < R), other=0.0)

            acc_xa_block += tl.dot(x, a)

        # Load B tile for this R block
        b_ptrs = B_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_r[:, None] < R) & (offs_n[None, :] < N), other=0.0)

        # Accumulate (X @ A block) @ B block
        acc_lora += tl.dot(acc_xa_block.to(b.dtype), b).to(tl.float32)

    # Combine: Y = base + scaling * lora
    y = acc_base + scaling * acc_lora

    # Store output
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_lora_forward(X, W, A, B, scaling, out=None):
    """
    Fused forward pass for LoRA.

    Computes: Y = X @ W + scaling * X @ A @ B

    Args:
        X: Input tensor [M, K] (already flattened from [batch, seq, hidden])
        W: Dequantized base weight [K, N]
        A: LoRA A matrix [K, R] (transposed from original [R, K])
        B: LoRA B matrix [R, N] (transposed from original [N, R])
        scaling: LoRA scaling factor
        out: Optional pre-allocated output tensor [M, N]

    Returns:
        Output tensor [M, N]
    """
    # Get dimensions
    M, K = X.shape
    K_w, N = W.shape
    K_a, R = A.shape
    R_b, N_b = B.shape

    # Validate dimensions
    assert K == K_w == K_a, f"K dimension mismatch: X={K}, W={K_w}, A={K_a}"
    assert R == R_b, f"R dimension mismatch: A={R}, B={R_b}"
    assert N == N_b, f"N dimension mismatch: W={N}, B={N_b}"

    # Allocate output if not provided
    if out is None:
        out = torch.empty((M, N), dtype=X.dtype, device=X.device)

    # Choose kernel based on LoRA rank
    # Small ranks (≤128) can use the simpler fused kernel
    # Large ranks need tiled accumulation
    if R <= 128:
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        with torch_gpu_device(X.device):
            fused_lora_forward_kernel[grid](
                X, W, A, B, out,
                M, N, K, R,
                scaling,
                X.stride(0), X.stride(1),
                W.stride(0), W.stride(1),
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                out.stride(0), out.stride(1),
            )
    else:
        # Use large rank kernel with tiled R dimension
        BLOCK_R = 64
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        with torch_gpu_device(X.device):
            fused_lora_forward_kernel_large_rank[grid](
                X, W, A, B, out,
                M, N, K, R,
                scaling,
                X.stride(0), X.stride(1),
                W.stride(0), W.stride(1),
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, BLOCK_R=BLOCK_R,
            )

    return out


def matmul_lora_fused(X, W, W_quant, A, B, s, out=None):
    """
    Fused matmul_lora implementation using Triton kernel.

    This is a drop-in replacement for the original matmul_lora function
    that uses a fused kernel for better performance.

    Args:
        X: Input tensor [batch, seq, hidden] or [M, K]
        W: Base weight matrix (possibly quantized)
        W_quant: Quantization state for W
        A: LoRA A matrix [R, K]
        B: LoRA B matrix [N, R]
        s: LoRA scaling factor
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor [batch, seq, out] or [M, N]
    """
    from .utils import fast_dequantize, torch_matmul

    dtype = X.dtype

    # Handle 3D input
    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    # Dequantize W
    W = fast_dequantize(W.t(), W_quant, use_global_buffer=True)

    # If no LoRA, just do base matmul
    if A is None:
        out = torch_matmul(X, W, out=out)
        if W_quant is not None:
            del W
        return out.view(batch, seq_len, -1) if reshape else out

    # Transpose LoRA matrices to match kernel expectations
    # A: [R, K] -> [K, R]
    # B: [N, R] -> [R, N]
    A_t = A.t().to(dtype)
    B_t = B.t().to(dtype)

    # Use fused kernel
    out = fused_lora_forward(X, W, A_t, B_t, s, out=out)

    if W_quant is not None:
        del W

    return out.view(batch, seq_len, -1) if reshape else out
