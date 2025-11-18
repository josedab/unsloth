"""
Unit tests for configuration handling.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestModelConfigValidation:
    """Tests for model configuration validation."""

    def test_model_config_required_fields(self, mock_model_config):
        """Test that model config has required fields."""
        required_fields = [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
        ]

        for field in required_fields:
            assert hasattr(mock_model_config, field), f"Missing required field: {field}"

    def test_model_config_hidden_size(self, mock_model_config):
        """Test model config hidden size."""
        assert mock_model_config.hidden_size > 0
        assert isinstance(mock_model_config.hidden_size, int)

    def test_model_config_num_heads(self, mock_model_config):
        """Test model config number of attention heads."""
        assert mock_model_config.num_attention_heads > 0
        assert isinstance(mock_model_config.num_attention_heads, int)

    def test_model_config_num_layers(self, mock_model_config):
        """Test model config number of layers."""
        assert mock_model_config.num_hidden_layers > 0
        assert isinstance(mock_model_config.num_hidden_layers, int)

    def test_model_config_vocab_size(self, mock_model_config):
        """Test model config vocabulary size."""
        assert mock_model_config.vocab_size > 0
        assert isinstance(mock_model_config.vocab_size, int)

    def test_model_config_max_position(self, mock_model_config):
        """Test model config max position embeddings."""
        assert mock_model_config.max_position_embeddings > 0

    def test_model_config_model_type(self, mock_model_config):
        """Test model config model type."""
        assert mock_model_config.model_type is not None
        assert isinstance(mock_model_config.model_type, str)


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_lora_config_r_value(self, lora_config):
        """Test LoRA config r value."""
        assert lora_config["r"] > 0
        assert isinstance(lora_config["r"], int)

    def test_lora_config_alpha(self, lora_config):
        """Test LoRA config alpha value."""
        assert lora_config["lora_alpha"] > 0

    def test_lora_config_dropout(self, lora_config):
        """Test LoRA config dropout."""
        assert 0 <= lora_config["lora_dropout"] <= 1

    def test_lora_config_target_modules(self, lora_config):
        """Test LoRA config target modules."""
        assert isinstance(lora_config["target_modules"], list)
        assert len(lora_config["target_modules"]) > 0

    def test_lora_config_bias(self, lora_config):
        """Test LoRA config bias setting."""
        assert lora_config["bias"] in ["none", "all", "lora_only"]

    def test_lora_config_task_type(self, lora_config):
        """Test LoRA config task type."""
        assert lora_config["task_type"] in ["CAUSAL_LM", "SEQ_2_SEQ_LM", "SEQ_CLS"]


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_training_args_batch_size(self, training_args):
        """Test training args batch size."""
        assert training_args["per_device_train_batch_size"] > 0

    def test_training_args_gradient_accumulation(self, training_args):
        """Test training args gradient accumulation."""
        assert training_args["gradient_accumulation_steps"] > 0

    def test_training_args_learning_rate(self, training_args):
        """Test training args learning rate."""
        assert 0 < training_args["learning_rate"] < 1

    def test_training_args_max_steps(self, training_args):
        """Test training args max steps."""
        assert training_args["max_steps"] > 0

    def test_training_args_seed(self, training_args):
        """Test training args seed."""
        assert isinstance(training_args["seed"], int)


class TestQuantizationConfig:
    """Tests for quantization configuration."""

    def test_4bit_config_structure(self):
        """Test 4-bit quantization config structure."""
        config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

        assert config["load_in_4bit"] is True
        assert config["bnb_4bit_quant_type"] in ["nf4", "fp4"]

    def test_8bit_config_structure(self):
        """Test 8-bit quantization config structure."""
        config = {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        }

        assert config["load_in_8bit"] is True


class TestSyntheticDataConfig:
    """Tests for synthetic data configuration."""

    def test_synthetic_configs_import(self):
        """Test that synthetic_configs can be imported."""
        try:
            from unsloth.dataprep.synthetic_configs import (
                ALPACA_CONFIG,
                SHAREGPT_CONFIG,
            )
            assert ALPACA_CONFIG is not None or SHAREGPT_CONFIG is not None
        except ImportError:
            pytest.skip("synthetic_configs not available")


class TestDeviceConfig:
    """Tests for device configuration."""

    def test_device_type_import(self):
        """Test that device_type module can be imported."""
        from unsloth import device_type
        assert device_type is not None


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_max_seq_length(self):
        """Test default max sequence length values."""
        common_lengths = [512, 1024, 2048, 4096, 8192]
        assert all(isinstance(l, int) for l in common_lengths)

    def test_default_lora_r_values(self):
        """Test common LoRA r values."""
        common_r = [4, 8, 16, 32, 64]
        assert all(isinstance(r, int) and r > 0 for r in common_r)

    def test_default_learning_rates(self):
        """Test common learning rates."""
        common_lr = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
        assert all(isinstance(lr, float) and 0 < lr < 1 for lr in common_lr)
