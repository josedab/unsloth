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
Tests for the fused LoRA forward pass kernel.

This module tests the numerical equivalence of the fused Triton kernel
against the standard PyTorch implementation.
"""

import pytest
import torch
import os

# Ensure fused forward is enabled for testing
os.environ.pop("UNSLOTH_DISABLE_FUSED_FORWARD", None)


def reference_lora_forward(X, W, A, B, scaling):
    """
    Reference PyTorch implementation of LoRA forward pass.

    Computes: Y = X @ W + scaling * X @ A @ B

    Args:
        X: Input tensor [M, K]
        W: Base weight [K, N]
        A: LoRA A matrix [K, R]
        B: LoRA B matrix [R, N]
        scaling: LoRA scaling factor

    Returns:
        Output tensor [M, N]
    """
    # Base matmul
    Y = torch.matmul(X, W)

    # LoRA contribution
    XA = torch.matmul(X, A)
    XAB = torch.matmul(XA, B)

    # Combine
    Y = Y + scaling * XAB

    return Y


@pytest.fixture
def device():
    """Get CUDA device if available, skip if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestFusedLoRAForward:
    """Test cases for fused LoRA forward kernel."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    @pytest.mark.parametrize("N", [64, 128, 256])
    @pytest.mark.parametrize("K", [64, 128, 256])
    @pytest.mark.parametrize("R", [8, 16, 32, 64])
    def test_numerical_equivalence_small_rank(self, device, dtype, M, N, K, R):
        """Test numerical equivalence for small LoRA ranks (R <= 128)."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        # Create input tensors
        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.5

        # Reference implementation
        Y_ref = reference_lora_forward(X, W, A, B, scaling)

        # Fused kernel
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        # Check numerical equivalence
        # Use slightly relaxed tolerance for fused kernels due to different accumulation order
        rtol = 1e-2 if dtype == torch.float16 else 5e-3
        atol = 1e-3 if dtype == torch.float16 else 1e-4

        assert torch.allclose(Y_ref, Y_fused, rtol=rtol, atol=atol), \
            f"Numerical mismatch: max diff = {(Y_ref - Y_fused).abs().max().item()}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("M,N,K,R", [
        (128, 256, 512, 128),
        (256, 512, 1024, 64),
        (64, 128, 256, 32),
    ])
    def test_numerical_equivalence_typical_sizes(self, device, dtype, M, N, K, R):
        """Test with typical model sizes."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 1.0

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        rtol = 1e-2 if dtype == torch.float16 else 5e-3
        atol = 1e-3 if dtype == torch.float16 else 1e-4

        assert torch.allclose(Y_ref, Y_fused, rtol=rtol, atol=atol), \
            f"Numerical mismatch: max diff = {(Y_ref - Y_fused).abs().max().item()}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_rank(self, device, dtype):
        """Test with large LoRA rank (R > 128) that uses tiled kernel."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 128, 256, 512, 256  # R > 128

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.25

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        rtol = 1e-2 if dtype == torch.float16 else 5e-3
        atol = 1e-3 if dtype == torch.float16 else 1e-4

        assert torch.allclose(Y_ref, Y_fused, rtol=rtol, atol=atol), \
            f"Numerical mismatch: max diff = {(Y_ref - Y_fused).abs().max().item()}"

    @pytest.mark.parametrize("scaling", [0.0, 0.5, 1.0, 2.0])
    def test_different_scaling_factors(self, device, scaling):
        """Test with different LoRA scaling factors."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 64, 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3), \
            f"Numerical mismatch with scaling={scaling}: max diff = {(Y_ref - Y_fused).abs().max().item()}"

    def test_output_buffer_reuse(self, device):
        """Test that pre-allocated output buffer is used correctly."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 64, 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.5

        # Pre-allocate output
        out = torch.empty(M, N, dtype=dtype, device=device)

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling, out=out)

        # Check that output buffer was used
        assert Y_fused.data_ptr() == out.data_ptr()

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3)

    def test_contiguous_tensors(self, device):
        """Test with contiguous tensors."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 64, 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device).contiguous()
        W = torch.randn(K, N, dtype=dtype, device=device).contiguous()
        A = torch.randn(K, R, dtype=dtype, device=device).contiguous()
        B = torch.randn(R, N, dtype=dtype, device=device).contiguous()
        scaling = 0.5

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3)

    @pytest.mark.parametrize("M", [1, 7, 15, 33, 65])
    def test_non_power_of_two_batch(self, device, M):
        """Test with non-power-of-2 batch sizes."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        N, K, R = 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.5

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3), \
            f"Numerical mismatch with M={M}: max diff = {(Y_ref - Y_fused).abs().max().item()}"


class TestMatmulLoraIntegration:
    """Test matmul_lora integration with fused kernel."""

    def test_matmul_lora_uses_fused_kernel(self, device):
        """Test that matmul_lora uses the fused kernel when enabled."""
        from unsloth.kernels.utils import matmul_lora, FUSED_LORA_FORWARD_ENABLED

        # Check that fused kernel is enabled
        assert FUSED_LORA_FORWARD_ENABLED, "Fused kernel should be enabled for testing"

        torch.manual_seed(42)
        M, N, K, R = 64, 128, 256, 16
        dtype = torch.float16

        # Create tensors in the format expected by matmul_lora
        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(N, K, dtype=dtype, device=device)  # Note: W is [N, K], will be transposed
        A = torch.randn(R, K, dtype=dtype, device=device)  # LoRA A: [R, K]
        B = torch.randn(N, R, dtype=dtype, device=device)  # LoRA B: [N, R]
        scaling = 0.5

        # Reference computation
        W_t = W.t()  # [K, N]
        A_t = A.t()  # [K, R]
        B_t = B.t()  # [R, N]
        Y_ref = reference_lora_forward(X, W_t, A_t, B_t, scaling)

        # matmul_lora computation (note: W_quant=None for non-quantized)
        Y_matmul = matmul_lora(X, W, None, A, B, scaling)

        assert torch.allclose(Y_ref, Y_matmul, rtol=1e-2, atol=1e-3), \
            f"matmul_lora mismatch: max diff = {(Y_ref - Y_matmul).abs().max().item()}"

    def test_matmul_lora_3d_input(self, device):
        """Test matmul_lora with 3D input [batch, seq, hidden]."""
        from unsloth.kernels.utils import matmul_lora

        torch.manual_seed(42)
        batch, seq_len, hidden = 2, 32, 256
        out_features = 128
        R = 16
        dtype = torch.float16

        X = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device)
        W = torch.randn(out_features, hidden, dtype=dtype, device=device)
        A = torch.randn(R, hidden, dtype=dtype, device=device)
        B = torch.randn(out_features, R, dtype=dtype, device=device)
        scaling = 0.5

        # Flatten X for reference
        X_flat = X.view(-1, hidden)
        W_t = W.t()
        A_t = A.t()
        B_t = B.t()
        Y_ref_flat = reference_lora_forward(X_flat, W_t, A_t, B_t, scaling)
        Y_ref = Y_ref_flat.view(batch, seq_len, out_features)

        Y_matmul = matmul_lora(X, W, None, A, B, scaling)

        assert Y_matmul.shape == (batch, seq_len, out_features)
        assert torch.allclose(Y_ref, Y_matmul, rtol=1e-2, atol=1e-3), \
            f"3D matmul_lora mismatch: max diff = {(Y_ref - Y_matmul).abs().max().item()}"

    def test_matmul_lora_no_lora(self, device):
        """Test matmul_lora without LoRA (A=None)."""
        from unsloth.kernels.utils import matmul_lora

        torch.manual_seed(42)
        M, N, K = 64, 128, 256
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(N, K, dtype=dtype, device=device)

        # Reference: just X @ W.t()
        Y_ref = torch.matmul(X, W.t())

        # matmul_lora with A=None
        Y_matmul = matmul_lora(X, W, None, None, None, 0.0)

        assert torch.allclose(Y_ref, Y_matmul, rtol=1e-3, atol=1e-4)


class TestEnvironmentVariableFallback:
    """Test environment variable fallback functionality."""

    def test_fallback_when_disabled(self, device):
        """Test that standard implementation is used when fused kernel is disabled."""
        import importlib

        # Save original environment
        original_env = os.environ.get("UNSLOTH_DISABLE_FUSED_FORWARD")

        try:
            # Disable fused kernel
            os.environ["UNSLOTH_DISABLE_FUSED_FORWARD"] = "1"

            # Reload module to pick up environment change
            import unsloth.kernels.utils as utils_module
            importlib.reload(utils_module)

            # Check that fused kernel is disabled
            assert not utils_module.FUSED_LORA_FORWARD_ENABLED

            # Test still works (uses standard implementation)
            torch.manual_seed(42)
            M, N, K, R = 64, 128, 256, 16
            dtype = torch.float16

            X = torch.randn(M, K, dtype=dtype, device=device)
            W = torch.randn(N, K, dtype=dtype, device=device)
            A = torch.randn(R, K, dtype=dtype, device=device)
            B = torch.randn(N, R, dtype=dtype, device=device)
            scaling = 0.5

            # Should still produce correct results
            Y = utils_module.matmul_lora(X, W, None, A, B, scaling)

            # Reference
            W_t = W.t()
            A_t = A.t()
            B_t = B.t()
            Y_ref = reference_lora_forward(X, W_t, A_t, B_t, scaling)

            assert torch.allclose(Y_ref, Y, rtol=1e-3, atol=1e-4)

        finally:
            # Restore original environment
            if original_env is None:
                os.environ.pop("UNSLOTH_DISABLE_FUSED_FORWARD", None)
            else:
                os.environ["UNSLOTH_DISABLE_FUSED_FORWARD"] = original_env

            # Reload to restore original state
            importlib.reload(utils_module)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self, device):
        """Test with single token (M=1)."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 1, 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.5

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3)

    def test_zero_scaling(self, device):
        """Test with zero scaling (no LoRA contribution)."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 64, 128, 256, 16
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.0

        # With zero scaling, result should be just X @ W
        Y_ref = torch.matmul(X, W)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-3, atol=1e-4)

    def test_small_dimensions(self, device):
        """Test with very small dimensions."""
        from unsloth.kernels.fused_lora_forward import fused_lora_forward

        torch.manual_seed(42)
        M, N, K, R = 4, 8, 16, 4
        dtype = torch.float16

        X = torch.randn(M, K, dtype=dtype, device=device)
        W = torch.randn(K, N, dtype=dtype, device=device)
        A = torch.randn(K, R, dtype=dtype, device=device)
        B = torch.randn(R, N, dtype=dtype, device=device)
        scaling = 0.5

        Y_ref = reference_lora_forward(X, W, A, B, scaling)
        Y_fused = fused_lora_forward(X, W, A, B, scaling)

        assert torch.allclose(Y_ref, Y_fused, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
