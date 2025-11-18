# RFC-0007: Fused LoRA Forward Pass

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 20 person-days
**Priority:** P2 (Long-term)

---

## Summary

Implement a fused Triton kernel for LoRA forward pass to match the optimization level of the backward pass, achieving 1.3-1.5x forward speedup and reducing the performance gap between forward and backward.

---

## Motivation

Unsloth's LoRA backward pass is highly optimized (5-10x speedup), but the forward pass uses standard PyTorch operations with multiple kernel launches and intermediate allocations:

```python
# Current forward pass - 4 separate operations
Y = matmul(X, W)                    # Op 1: Base computation
XA = matmul(X, A)                   # Op 2: LoRA first matmul
XAB = matmul(XA, B)                 # Op 3: LoRA second matmul
Y = Y + scaling * XAB               # Op 4: Combine
```

Each operation requires:
- Kernel launch overhead (~5-10µs each)
- Global memory read/write
- Intermediate tensor allocation

### Performance Gap

| Pass | Current | With Optimization | Gap |
|------|---------|-------------------|-----|
| Forward | 1.0x | 1.3-1.5x | 30-50% |
| Backward | 5-10x | 5-10x | Optimized |

### Impact

For a typical training step:
- Forward pass: ~40% of compute time
- Backward pass: ~60% of compute time

A 1.3x forward speedup → ~12% overall training speedup.

---

## Detailed Design

### Fused Kernel Approach

Combine all forward operations into a single kernel:

```python
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
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ak, stride_ar,
    stride_br, stride_bn,
    stride_ym, stride_yn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y = X @ W + scaling * X @ A @ B

    Optimization strategy:
    1. Load X tile once, use for both W and A matmuls
    2. Accumulate results in registers
    3. Single store to global memory
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

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X tile (shared between base and LoRA)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Base matmul: X @ W
        w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc_base += tl.dot(x, w)

        # LoRA matmul: X @ A
        a_ptrs = A_ptr + offs_k[:, None] * stride_ak + tl.arange(0, R)[None, :] * stride_ar
        a = tl.load(a_ptrs, mask=offs_k[:, None] < K, other=0.0)
        xa = tl.dot(x, a)  # [BLOCK_M, R]

        # Continue with (X @ A) @ B in same iteration if R is small
        # Otherwise accumulate XA and do second matmul after K loop

    # Second matmul for LoRA if R > BLOCK_K
    # ... (implementation depends on R size)

    # Combine: Y = base + scaling * lora
    y = acc_base + scaling * acc_lora

    # Store output
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### Integration with Existing Code

```python
# unsloth/kernels/fast_lora.py

class LoRA_MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, W_quant, A, B, scaling):
        # Save for backward
        ctx.save_for_backward(X, W_quant, A, B)
        ctx.scaling = scaling

        # Use fused kernel
        Y = fused_lora_forward(X, W, W_quant, A, B, scaling)

        return Y

    @staticmethod
    def backward(ctx, dY):
        # Existing optimized backward
        ...


def fused_lora_forward(X, W, W_quant, A, B, scaling):
    """
    Fused forward pass for LoRA.

    Args:
        X: Input tensor [batch, seq, hidden]
        W: Base weight (4-bit quantized)
        W_quant: Quantization state
        A: LoRA A matrix [hidden, rank]
        B: LoRA B matrix [rank, out]
        scaling: LoRA scaling factor

    Returns:
        Output tensor [batch, seq, out]
    """
    # Reshape for matmul
    batch, seq, hidden = X.shape
    X_flat = X.view(-1, hidden)
    out_features = B.shape[1]

    # Output tensor
    Y = torch.empty(batch * seq, out_features, dtype=X.dtype, device=X.device)

    # Dequantize W (or use quantized kernel)
    W_deq = fast_dequantize(W, W_quant)

    # Launch fused kernel
    M, N, K = X_flat.shape[0], out_features, hidden
    R = A.shape[1]

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_lora_forward_kernel[grid](
        X_flat, W_deq, A, B, Y,
        M, N, K, R,
        scaling,
        X_flat.stride(0), X_flat.stride(1),
        W_deq.stride(0), W_deq.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    return Y.view(batch, seq, out_features)
```

### Autotuning Configuration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_lora_forward_kernel(...):
    ...
```

---

## Example Usage

No API changes required. Users automatically get the optimization:

```python
from unsloth import FastLanguageModel

# Load model (forward now uses fused kernel)
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Apply LoRA (forward now uses fused kernel)
model = FastLanguageModel.get_peft_model(model, r=16)

# Training (forward automatically optimized)
trainer = UnslothTrainer(model=model, ...)
trainer.train()
```

### Verifying Optimization

```python
# Check kernel is being used
from unsloth.kernels.fast_lora import FUSED_FORWARD_ENABLED
print(f"Fused forward: {FUSED_FORWARD_ENABLED}")  # True
```

---

## Implementation Plan

### Phase 1: Kernel Development (Days 1-8)

**Days 1-3: Basic Kernel**
- Implement unfused version for correctness
- Add basic fused kernel (no quantization)
- Verify numerical correctness

**Days 4-6: Quantization Support**
- Handle 4-bit weight dequantization
- Fuse dequantization into kernel
- Verify correctness with quantized weights

**Days 7-8: Edge Cases**
- Handle different LoRA ranks
- Handle different batch sizes
- Add boundary checks

### Phase 2: Optimization (Days 9-14)

**Days 9-10: Autotuning**
- Add autotuning configurations
- Benchmark different block sizes
- Optimize for common dimensions

**Days 11-12: Memory Optimization**
- Minimize shared memory usage
- Optimize memory access patterns
- Add prefetching

**Days 13-14: Performance Validation**
- Benchmark against PyTorch
- Test across different GPUs
- Profile and optimize hotspots

### Phase 3: Integration (Days 15-18)

**Days 15-16: Integration**
- Replace forward in LoRA_MLP class
- Add to LoRA_QKV class
- Update all LoRA modules

**Days 17-18: Testing**
- Run existing tests
- Add kernel-specific tests
- Verify no regressions

### Phase 4: Documentation (Days 19-20)

**Day 19: Documentation**
- Document kernel implementation
- Add performance notes
- Update contributor guide

**Day 20: Final Review**
- Code review
- Final benchmarking
- Merge preparation

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | Basic kernel working |
| 8 | Quantization support |
| 14 | Performance optimized |
| 18 | Fully integrated |
| 20 | Ready for merge |

---

## Backwards Compatibility

### Breaking Changes

None. Internal optimization only.

### Fallback

```python
# Environment variable to disable
if os.environ.get("UNSLOTH_DISABLE_FUSED_FORWARD"):
    # Use original implementation
    Y = matmul(X, W) + scaling * matmul(matmul(X, A), B)
else:
    # Use fused kernel
    Y = fused_lora_forward(X, W, A, B, scaling)
```

### Numerical Precision

Must match PyTorch exactly (within floating-point tolerance):

```python
def test_numerical_equivalence():
    Y_pytorch = matmul(X, W) + scaling * matmul(matmul(X, A), B)
    Y_fused = fused_lora_forward(X, W, A, B, scaling)

    assert torch.allclose(Y_pytorch, Y_fused, rtol=1e-5, atol=1e-5)
```

---

## Alternatives Considered

### Alternative 1: Torch.compile

Use PyTorch's compiler:

```python
@torch.compile
def lora_forward(X, W, A, B, scaling):
    return matmul(X, W) + scaling * matmul(matmul(X, A), B)
```

**Rejected:**
- Less control over optimization
- Compilation overhead
- May not fuse optimally

### Alternative 2: CUTLASS Kernels

Use NVIDIA CUTLASS instead of Triton:

```cpp
// CUTLASS kernel implementation
```

**Rejected:**
- More complex to implement
- Less portable
- Triton is sufficient

### Alternative 3: Optimize Memory Access Only

Keep separate kernels, optimize memory:

```python
# Prefetch and reuse X
X_cached = X.contiguous()
Y = matmul(X_cached, W)
XA = matmul(X_cached, A)
...
```

**Rejected:**
- Still multiple kernel launches
- Limited speedup potential

---

## Open Questions

1. **How to handle varying LoRA ranks?**
   - Small R: Fuse second matmul
   - Large R: Separate accumulation
   - Threshold TBD

2. **Shared memory limitations?**
   - Some GPUs have limited shared memory
   - May need fallback for older GPUs

3. **Multi-GPU considerations?**
   - Kernel should work with DDP
   - Test with tensor parallelism

4. **FP8 support?**
   - H100 has FP8 tensor cores
   - Future optimization opportunity

---

## Success Criteria

- [ ] 1.3x minimum forward pass speedup
- [ ] Numerical equivalence to PyTorch (rtol=1e-5)
- [ ] No memory regression
- [ ] Works with all quantization types (4-bit, 8-bit, 16-bit)
- [ ] All existing tests pass
- [ ] Performance consistent across GPU types (T4, A100, H100)

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Forward pass time | 1.0x | 1.3-1.5x |
| Kernel launches per forward | 4 | 1 |
| Intermediate allocations | 2 | 0 |
| Overall training speedup | - | ~12% |

---

## Required Approvals

- [ ] Kernel correctness review
- [ ] Performance benchmarking approval
- [ ] GPU compatibility testing

---

## Rollback Strategy

1. Environment variable to disable fused kernel
2. Fallback to original implementation
3. No data format changes

---

## References

- Triton Documentation: https://triton-lang.org/
- CUTLASS Efficient GEMM: https://github.com/NVIDIA/cutlass
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- Existing fast_lora.py implementation

---

*Next: [RFC-0008: Multi-GPU Optimization](./RFC-0008-multi-gpu.md)*
