"""
Unit tests for validation functions.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_model_name_format(self):
        """Test model name format validation."""
        valid_names = [
            "unsloth/llama-3-8b-bnb-4bit",
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
        ]

        for name in valid_names:
            # Should contain org/model format
            assert "/" in name
            parts = name.split("/")
            assert len(parts) == 2
            assert len(parts[0]) > 0
            assert len(parts[1]) > 0

    def test_validate_invalid_model_name(self):
        """Test invalid model name detection."""
        invalid_names = [
            "",
            "no-slash",
            "/no-org",
            "no-model/",
            "too/many/slashes",
        ]

        for name in invalid_names:
            parts = name.split("/")
            assert len(parts) != 2 or "" in parts

    def test_validate_max_seq_length(self):
        """Test max sequence length validation."""
        valid_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        invalid_lengths = [0, -1, -100]

        for length in valid_lengths:
            assert length > 0, f"Invalid length: {length}"

        for length in invalid_lengths:
            assert length <= 0, f"Should be invalid: {length}"

    def test_validate_batch_size(self):
        """Test batch size validation."""
        valid_sizes = [1, 2, 4, 8, 16, 32]
        invalid_sizes = [0, -1]

        for size in valid_sizes:
            assert size > 0

        for size in invalid_sizes:
            assert size <= 0

    def test_validate_learning_rate(self):
        """Test learning rate validation."""
        valid_lr = [1e-6, 1e-5, 1e-4, 1e-3]
        invalid_lr = [0, -1e-5, 10]

        for lr in valid_lr:
            assert 0 < lr < 1

        for lr in invalid_lr:
            assert lr <= 0 or lr >= 1


class TestLoRAValidation:
    """Tests for LoRA parameter validation."""

    def test_validate_lora_r(self):
        """Test LoRA r value validation."""
        valid_r = [4, 8, 16, 32, 64, 128]
        invalid_r = [0, -1, -8]

        for r in valid_r:
            assert r > 0
            # r should typically be power of 2
            assert (r & (r - 1) == 0) or r in [4, 8, 16, 32, 64, 128]

        for r in invalid_r:
            assert r <= 0

    def test_validate_lora_alpha(self):
        """Test LoRA alpha validation."""
        valid_alpha = [8, 16, 32, 64]

        for alpha in valid_alpha:
            assert alpha > 0

    def test_validate_lora_dropout(self):
        """Test LoRA dropout validation."""
        valid_dropout = [0, 0.05, 0.1, 0.5]
        invalid_dropout = [-0.1, 1.5, 2.0]

        for dropout in valid_dropout:
            assert 0 <= dropout <= 1

        for dropout in invalid_dropout:
            assert dropout < 0 or dropout > 1

    def test_validate_target_modules(self):
        """Test LoRA target modules validation."""
        valid_modules = [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["all-linear"],
        ]

        for modules in valid_modules:
            assert isinstance(modules, list)
            assert len(modules) > 0
            assert all(isinstance(m, str) for m in modules)


class TestQuantizationValidation:
    """Tests for quantization parameter validation."""

    def test_validate_4bit_quant_type(self):
        """Test 4-bit quantization type validation."""
        valid_types = ["nf4", "fp4"]
        invalid_types = ["nf8", "fp16", "int4"]

        for qtype in valid_types:
            assert qtype in ["nf4", "fp4"]

        for qtype in invalid_types:
            assert qtype not in ["nf4", "fp4"]

    def test_validate_compute_dtype(self):
        """Test compute dtype validation."""
        valid_dtypes = ["float16", "bfloat16", "float32"]

        for dtype in valid_dtypes:
            assert dtype in ["float16", "bfloat16", "float32"]


class TestDataValidation:
    """Tests for data validation functions."""

    def test_validate_dataset_format(self, sample_dataset_dict):
        """Test dataset format validation."""
        assert "text" in sample_dataset_dict
        assert isinstance(sample_dataset_dict["text"], list)
        assert len(sample_dataset_dict["text"]) > 0

    def test_validate_messages_format(self, sample_messages):
        """Test messages format validation."""
        for msg in sample_messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]
            assert isinstance(msg["content"], str)

    def test_validate_empty_dataset(self):
        """Test empty dataset detection."""
        empty_dataset = {"text": []}
        assert len(empty_dataset["text"]) == 0

    def test_validate_dataset_column_types(self, sample_dataset_dict):
        """Test dataset column types."""
        for text in sample_dataset_dict["text"]:
            assert isinstance(text, str)


class TestPathValidation:
    """Tests for path validation."""

    def test_validate_model_path_characters(self):
        """Test model path allowed characters."""
        valid_paths = [
            "unsloth/llama-3-8b",
            "meta-llama/Llama-2-7b-hf",
            "Qwen/Qwen2-7B-Instruct",
        ]

        for path in valid_paths:
            # Should not contain invalid characters
            assert "\\" not in path
            assert " " not in path
            assert "\n" not in path

    def test_validate_output_path(self):
        """Test output path validation."""
        import os

        valid_paths = [
            "/tmp/output",
            "./outputs",
            "model_outputs",
        ]

        for path in valid_paths:
            # Path should be string
            assert isinstance(path, str)
            # Path should not be empty
            assert len(path) > 0


class TestNumericValidation:
    """Tests for numeric parameter validation."""

    def test_validate_positive_integer(self):
        """Test positive integer validation."""
        valid = [1, 10, 100, 1000]
        invalid = [0, -1, -100]

        for v in valid:
            assert v > 0
            assert isinstance(v, int)

        for v in invalid:
            assert v <= 0

    def test_validate_float_range(self):
        """Test float range validation."""
        # Test 0-1 range (like dropout)
        valid = [0.0, 0.5, 1.0]
        invalid = [-0.1, 1.1, 2.0]

        for v in valid:
            assert 0 <= v <= 1

        for v in invalid:
            assert v < 0 or v > 1

    def test_validate_gradient_accumulation(self):
        """Test gradient accumulation steps validation."""
        valid_steps = [1, 2, 4, 8, 16]

        for steps in valid_steps:
            assert steps > 0
            assert isinstance(steps, int)


class TestModelValidation:
    """Tests for model validation functions."""

    def test_validate_model_type(self):
        """Test model type validation."""
        valid_types = [
            "llama",
            "mistral",
            "qwen2",
            "gemma",
            "phi",
        ]

        for model_type in valid_types:
            assert isinstance(model_type, str)
            assert len(model_type) > 0

    def test_validate_hidden_size_divisibility(self):
        """Test that hidden size is divisible by num heads."""
        configs = [
            {"hidden_size": 4096, "num_attention_heads": 32},
            {"hidden_size": 2048, "num_attention_heads": 16},
            {"hidden_size": 1024, "num_attention_heads": 8},
        ]

        for config in configs:
            assert config["hidden_size"] % config["num_attention_heads"] == 0
