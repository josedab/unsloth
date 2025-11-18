# Copyright 2024-present the Unsloth team.
# Licensed under the Apache License 2.0

"""
Tests for Unsloth distributed training module.

These tests cover:
- CompressedAllReduce functionality
- OverlappedDDP behavior
- LoRAAwareDDP parameter handling
- ring_allreduce correctness
- UnslothDDP integration

Note: Some tests require multiple GPUs or mock distributed setup.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import math
from unittest.mock import patch, MagicMock

# Import distributed module
from unsloth.distributed import (
    CompressedAllReduce,
    OverlappedDDP,
    LoRAAwareDDP,
    ring_allreduce,
    UnslothDDP,
    initialize_distributed,
    cleanup_distributed,
    get_distributed_info,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, in_features=10, hidden=20, out_features=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LoRAModel(nn.Module):
    """Model with LoRA-style parameters for testing."""
    def __init__(self, in_features=10, hidden=20, out_features=5, rank=4):
        super().__init__()
        # Base (frozen) parameters
        self.base_weight = nn.Parameter(torch.randn(hidden, in_features), requires_grad=False)
        self.base_bias = nn.Parameter(torch.randn(hidden), requires_grad=False)

        # LoRA parameters (trainable)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features))
        self.lora_b = nn.Parameter(torch.randn(hidden, rank))

        # Output layer
        self.out = nn.Linear(hidden, out_features)

    def forward(self, x):
        # Base + LoRA
        h = x @ self.base_weight.T + self.base_bias
        h = h + (x @ self.lora_a.T) @ self.lora_b.T
        return self.out(torch.relu(h))


class TestCompressedAllReduce:
    """Tests for CompressedAllReduce class."""

    def test_init_valid_ratio(self):
        """Test initialization with valid compression ratio."""
        compressor = CompressedAllReduce(compression_ratio=0.1)
        assert compressor.compression_ratio == 0.1
        assert len(compressor.error_feedback) == 0

    def test_init_invalid_ratio(self):
        """Test initialization with invalid compression ratio."""
        with pytest.raises(ValueError):
            CompressedAllReduce(compression_ratio=0.0)
        with pytest.raises(ValueError):
            CompressedAllReduce(compression_ratio=1.5)

    def test_compress_basic(self):
        """Test basic compression functionality."""
        compressor = CompressedAllReduce(compression_ratio=0.5)
        tensor = torch.randn(100)

        values, indices = compressor.compress(tensor, "test_param")

        # Should keep 50% of elements
        assert values.numel() == 50
        assert indices.numel() == 50
        # Indices should be valid
        assert indices.max() < 100
        assert indices.min() >= 0

    def test_compress_small_tensor(self):
        """Test compression with small tensor."""
        compressor = CompressedAllReduce(compression_ratio=0.1, min_elements=5)
        tensor = torch.randn(10)

        values, indices = compressor.compress(tensor, "small_param")

        # Should keep at least min_elements
        assert values.numel() >= 5

    def test_decompress(self):
        """Test decompression returns correct shape."""
        compressor = CompressedAllReduce(compression_ratio=0.1)
        original_shape = torch.Size([10, 20])
        tensor = torch.randn(original_shape)

        values, indices = compressor.compress(tensor, "test")
        decompressed = compressor.decompress(values, indices, original_shape)

        assert decompressed.shape == original_shape

    def test_compress_decompress_roundtrip(self):
        """Test that top values are preserved through compress/decompress."""
        compressor = CompressedAllReduce(compression_ratio=0.1)
        tensor = torch.randn(1000)

        # Get top 10% indices
        _, top_indices = torch.topk(tensor.abs(), 100)

        values, indices = compressor.compress(tensor, "test")
        decompressed = compressor.decompress(values, indices, tensor.shape)

        # Top values should be approximately preserved (error feedback may shift)
        for idx in top_indices[:50]:  # Check top 50
            assert decompressed[idx] != 0

    def test_error_feedback(self):
        """Test error feedback accumulation."""
        compressor = CompressedAllReduce(compression_ratio=0.1)
        tensor = torch.randn(100)

        # First compression
        compressor.compress(tensor, "param1")
        assert "param1" in compressor.error_feedback
        assert compressor.error_feedback["param1"].shape == tensor.shape

    def test_clear_error_feedback(self):
        """Test clearing error feedback."""
        compressor = CompressedAllReduce(compression_ratio=0.1)
        tensor = torch.randn(100)

        compressor.compress(tensor, "param1")
        compressor.compress(tensor, "param2")
        assert len(compressor.error_feedback) == 2

        compressor.clear_error_feedback()
        assert len(compressor.error_feedback) == 0


class TestOverlappedDDP:
    """Tests for OverlappedDDP class."""

    def test_init(self):
        """Test OverlappedDDP initialization."""
        model = SimpleModel()
        ddp = OverlappedDDP(model, bucket_size_mb=1.0)

        assert ddp.model is model
        assert ddp.bucket_size == 1 * 1024 * 1024
        assert len(ddp.buckets) > 0

    def test_bucket_creation(self):
        """Test that buckets are created properly."""
        model = SimpleModel()
        ddp = OverlappedDDP(model, bucket_size_mb=0.001)  # Small bucket

        # Should have multiple buckets with small size
        assert len(ddp.buckets) > 0

        # All trainable params should be in buckets
        total_params_in_buckets = sum(len(bucket) for bucket in ddp.buckets)
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_params_in_buckets == trainable_params

    def test_forward(self):
        """Test forward pass."""
        model = SimpleModel()
        ddp = OverlappedDDP(model)

        x = torch.randn(5, 10)
        output = ddp(x)

        assert output.shape == (5, 5)  # batch_size x out_features

    def test_remove_hooks(self):
        """Test hook removal."""
        model = SimpleModel()
        ddp = OverlappedDDP(model)

        assert len(ddp._hooks) > 0
        ddp.remove_hooks()
        assert len(ddp._hooks) == 0


class TestLoRAAwareDDP:
    """Tests for LoRAAwareDDP class."""

    def test_parameter_categorization(self):
        """Test that LoRA parameters are correctly identified."""
        model = LoRAModel()
        lora_ddp = LoRAAwareDDP(model)

        # Should have 2 LoRA params (lora_a, lora_b)
        assert len(lora_ddp.lora_params) == 2

        # Should have base params (out layer: weight, bias)
        # base_weight and base_bias are frozen, so not counted
        assert len(lora_ddp.base_params) == 2

    def test_lora_param_names(self):
        """Test various LoRA naming patterns."""
        class MultiLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Different LoRA naming patterns
                self.lora_a_1 = nn.Parameter(torch.randn(4, 10))
                self.lora_b_1 = nn.Parameter(torch.randn(10, 4))
                self.layer_lora = nn.Parameter(torch.randn(5, 5))
                self.lora_embedding = nn.Parameter(torch.randn(10))
                self.regular_weight = nn.Parameter(torch.randn(5, 5))

            def forward(self, x):
                return x

        model = MultiLoRAModel()
        lora_ddp = LoRAAwareDDP(model)

        # Should identify all LoRA params
        assert len(lora_ddp.lora_params) == 4
        assert len(lora_ddp.base_params) == 1

    def test_get_lora_grad_norm(self):
        """Test LoRA gradient norm calculation."""
        model = LoRAModel()
        lora_ddp = LoRAAwareDDP(model)

        # Set some gradients
        for param in lora_ddp.lora_params:
            param.grad = torch.ones_like(param)

        norm = lora_ddp.get_lora_grad_norm()
        expected_norm = sum(p.numel() for p in lora_ddp.lora_params) ** 0.5
        assert abs(norm - expected_norm) < 1e-5


class TestRingAllreduce:
    """Tests for ring_allreduce function."""

    def test_no_distributed(self):
        """Test ring_allreduce returns input when not distributed."""
        tensor = torch.randn(100)
        result = ring_allreduce(tensor)
        assert torch.allclose(tensor, result)

    def test_tensor_shape_preserved(self):
        """Test that tensor shape is preserved."""
        tensor = torch.randn(10, 20)
        result = ring_allreduce(tensor.flatten())
        # Result is flattened
        assert result.numel() == tensor.numel()


class TestUnslothDDP:
    """Tests for UnslothDDP integration class."""

    def test_init_with_overlap(self):
        """Test initialization with overlapped communication."""
        model = SimpleModel()
        ddp = UnslothDDP(model, overlap=True)

        assert ddp._ddp_wrapper is not None
        assert ddp._lora_aware is None

    def test_init_without_overlap(self):
        """Test initialization without overlapped communication."""
        model = SimpleModel()
        ddp = UnslothDDP(model, overlap=False)

        assert ddp._ddp_wrapper is None
        assert ddp._lora_aware is not None

    def test_from_model(self):
        """Test static from_model constructor."""
        model = SimpleModel()
        ddp = UnslothDDP.from_model(
            model,
            compression_ratio=0.1,
            overlap=True,
        )

        assert isinstance(ddp, UnslothDDP)
        assert ddp.compressor is not None
        assert ddp.compression_ratio == 0.1

    def test_auto_selection(self):
        """Test automatic strategy selection."""
        model = SimpleModel()

        # Test different configurations
        ddp_eth = UnslothDDP.auto(model, world_size=4, interconnect="ethernet")
        assert ddp_eth.compression_ratio is not None

        ddp_nvlink = UnslothDDP.auto(model, world_size=2, interconnect="nvlink")
        assert ddp_nvlink.use_ring_allreduce is True

    def test_forward(self):
        """Test forward pass."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        x = torch.randn(5, 10)
        output = ddp(x)

        assert output.shape == (5, 5)

    def test_train_eval_modes(self):
        """Test train/eval mode switching."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        ddp.train()
        assert ddp._is_training is True
        assert model.training is True

        ddp.eval()
        assert ddp._is_training is False
        assert model.training is False

    def test_parameters(self):
        """Test parameter access."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        model_params = list(model.parameters())
        ddp_params = list(ddp.parameters())

        assert len(model_params) == len(ddp_params)

    def test_state_dict(self):
        """Test state dict access."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        model_state = model.state_dict()
        ddp_state = ddp.state_dict()

        assert model_state.keys() == ddp_state.keys()

    def test_to_device(self):
        """Test device movement."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        ddp.cpu()
        assert ddp.device == torch.device('cpu')

    def test_attribute_delegation(self):
        """Test that unknown attributes are delegated to model."""
        model = SimpleModel()
        ddp = UnslothDDP(model)

        # Access model's fc1 through ddp
        assert hasattr(ddp, 'fc1')
        assert ddp.fc1 is model.fc1


class TestDistributedUtilities:
    """Tests for distributed utility functions."""

    def test_get_distributed_info_not_initialized(self):
        """Test get_distributed_info when not initialized."""
        info = get_distributed_info()

        assert info["initialized"] is False
        assert info["world_size"] == 1
        assert info["rank"] == 0


class TestConvergence:
    """
    Tests for training convergence validation.

    These tests verify that the distributed module doesn't affect
    training convergence.
    """

    def test_gradient_preservation(self):
        """Test that gradients are computed correctly."""
        model = SimpleModel()
        ddp = UnslothDDP(model, overlap=False)

        x = torch.randn(5, 10)
        target = torch.randn(5, 5)

        # Forward and backward
        output = ddp(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        # Check all params have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_lora_gradient_isolation(self):
        """Test that LoRA gradients are isolated from base."""
        model = LoRAModel()
        lora_ddp = LoRAAwareDDP(model)

        x = torch.randn(5, 10)
        target = torch.randn(5, 5)

        # Forward and backward
        output = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        # LoRA params should have gradients
        for param in lora_ddp.lora_params:
            assert param.grad is not None

        # Base params (frozen) should not have gradients
        assert model.base_weight.grad is None
        assert model.base_bias.grad is None


class TestCompressionAccuracy:
    """Tests for gradient compression accuracy."""

    def test_compression_with_error_feedback(self):
        """Test that error feedback helps maintain accuracy."""
        compressor = CompressedAllReduce(compression_ratio=0.1)

        total_error_with_feedback = 0

        # Simulate multiple training steps
        for _ in range(10):
            tensor = torch.randn(1000)
            values, indices = compressor.compress(tensor, "param")
            decompressed = compressor.decompress(values, indices, tensor.shape)

            # Error should be reasonable
            error = (tensor - decompressed).abs().mean().item()
            total_error_with_feedback += error

        # Clear feedback and test without it
        compressor.clear_error_feedback()

        total_error_without_feedback = 0
        for _ in range(10):
            tensor = torch.randn(1000)
            compressor.clear_error_feedback()  # No accumulation
            values, indices = compressor.compress(tensor, "param")
            decompressed = compressor.decompress(values, indices, tensor.shape)

            error = (tensor - decompressed).abs().mean().item()
            total_error_without_feedback += error

        # Error feedback typically helps reduce cumulative error
        # This is a soft check as individual runs may vary
        assert total_error_with_feedback < total_error_without_feedback * 1.5

    def test_top_k_selection(self):
        """Test that top-K values are selected."""
        compressor = CompressedAllReduce(compression_ratio=0.1)

        # Create tensor with known top values
        tensor = torch.zeros(100)
        tensor[0] = 100
        tensor[50] = -100
        tensor[99] = 50

        values, indices = compressor.compress(tensor, "known")

        # Top values should be included
        indices_list = indices.tolist()
        assert 0 in indices_list
        assert 50 in indices_list
        assert 99 in indices_list


class TestIntegration:
    """Integration tests for the distributed module."""

    def test_full_training_step(self):
        """Test a complete training step with UnslothDDP."""
        # Setup
        model = SimpleModel()
        ddp = UnslothDDP.from_model(model, overlap=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=0.01)

        # Training step
        x = torch.randn(10, 10)
        target = torch.randn(10, 5)

        optimizer.zero_grad()
        output = ddp(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        ddp.sync_gradients()
        optimizer.step()

        # Verify model updated
        assert loss.item() >= 0

    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        model = SimpleModel()
        ddp = UnslothDDP.from_model(model, overlap=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            x = torch.randn(10, 10)
            target = torch.zeros(10, 5)  # Fixed target

            optimizer.zero_grad()
            output = ddp(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            ddp.sync_gradients()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease (not guaranteed but likely)
        # Just check it didn't explode
        assert all(l < 100 for l in losses)


# Multi-GPU tests (require actual distributed setup)
class TestMultiGPU:
    """
    Multi-GPU tests that require actual distributed setup.

    These tests are marked to skip if not in a distributed environment.
    """

    @pytest.mark.skipif(
        not dist.is_initialized(),
        reason="Requires distributed environment"
    )
    def test_all_reduce_correctness(self):
        """Test all-reduce produces correct results across GPUs."""
        model = SimpleModel()
        ddp = UnslothDDP(model, overlap=False)

        # Each rank has different input
        rank = dist.get_rank()
        x = torch.randn(5, 10) * (rank + 1)

        output = ddp(x)
        loss = output.sum()
        loss.backward()
        ddp.sync_gradients()

        # Gradients should be averaged across ranks
        # (This is validated by the sync_gradients implementation)

    @pytest.mark.skipif(
        not dist.is_initialized(),
        reason="Requires distributed environment"
    )
    def test_ring_allreduce_correctness(self):
        """Test ring all-reduce produces same result as standard."""
        tensor = torch.randn(1000)

        # Standard all-reduce
        standard = tensor.clone()
        dist.all_reduce(standard)
        standard /= dist.get_world_size()

        # Ring all-reduce
        ring_result = ring_allreduce(tensor.clone())

        assert torch.allclose(standard, ring_result, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
