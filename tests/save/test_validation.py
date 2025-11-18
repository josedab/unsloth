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

"""Tests for unsloth.save.validation module."""

import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import MagicMock, patch

from unsloth.save.validation import (
    validate_model,
    validate_tokenizer,
    validate_directory,
    validate_hub_token,
    validate_quantization_method,
    validate_save_method,
)
from unsloth.save.base import ValidationError


class TestValidateModel:
    """Tests for validate_model function."""

    def test_none_model_raises_error(self):
        """Test that None model raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_model(None)
        assert "cannot be None" in str(exc_info.value)

    def test_model_without_config_raises_error(self):
        """Test that model without config raises ValidationError."""
        model = MagicMock(spec=[])
        with pytest.raises(ValidationError) as exc_info:
            validate_model(model)
        assert "config" in str(exc_info.value)

    def test_valid_model_passes(self):
        """Test that valid model passes validation."""
        model = MagicMock()
        model.config = MagicMock()
        model.state_dict = MagicMock()
        # Should not raise
        validate_model(model)


class TestValidateTokenizer:
    """Tests for validate_tokenizer function."""

    def test_none_tokenizer_not_required(self):
        """Test that None tokenizer passes when not required."""
        # Should not raise
        validate_tokenizer(None, require=False)

    def test_none_tokenizer_required(self):
        """Test that None tokenizer raises when required."""
        with pytest.raises(ValidationError) as exc_info:
            validate_tokenizer(None, require=True)
        assert "required" in str(exc_info.value)

    def test_valid_tokenizer_passes(self):
        """Test that valid tokenizer passes validation."""
        tokenizer = MagicMock()
        tokenizer.save_pretrained = MagicMock()
        # Should not raise
        validate_tokenizer(tokenizer)


class TestValidateDirectory:
    """Tests for validate_directory function."""

    def test_creates_directory(self):
        """Test that directory is created when create=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "new_dir")
            result = validate_directory(test_dir, create=True)
            assert result.exists()
            assert result.is_dir()

    def test_returns_path_object(self):
        """Test that Path object is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_directory(tmpdir)
            assert isinstance(result, Path)

    def test_nonexistent_dir_raises_when_not_creating(self):
        """Test that nonexistent dir raises error when create=False."""
        with pytest.raises(ValidationError):
            validate_directory("/nonexistent/path/123456", create=False)


class TestValidateQuantizationMethod:
    """Tests for validate_quantization_method function."""

    def test_valid_methods(self):
        """Test that valid quantization methods pass."""
        assert validate_quantization_method("q4_k_m") == "q4_k_m"
        assert validate_quantization_method("q8_0") == "q8_0"
        assert validate_quantization_method("f16") == "f16"

    def test_special_aliases(self):
        """Test that special aliases are accepted."""
        assert validate_quantization_method("fast_quantized") == "fast_quantized"
        assert validate_quantization_method("quantized") == "quantized"
        assert validate_quantization_method("not_quantized") == "not_quantized"

    def test_case_insensitive(self):
        """Test that method validation is case-insensitive."""
        assert validate_quantization_method("Q4_K_M") == "q4_k_m"
        assert validate_quantization_method("F16") == "f16"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_quantization_method("invalid_method")
        assert "Invalid quantization method" in str(exc_info.value)


class TestValidateSaveMethod:
    """Tests for validate_save_method function."""

    def test_valid_methods(self):
        """Test that valid save methods pass."""
        assert validate_save_method("lora") == "lora"
        assert validate_save_method("merged_16bit") == "merged_16bit"
        assert validate_save_method("merged_4bit") == "merged_4bit"

    def test_normalized_format(self):
        """Test that methods are normalized."""
        assert validate_save_method("LORA") == "lora"
        assert validate_save_method("merged 16bit") == "merged_16bit"

    def test_forced_variant(self):
        """Test that merged_4bit_forced is converted."""
        assert validate_save_method("merged_4bit_forced") == "merged_4bit"

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_save_method("invalid")
        assert "Invalid save method" in str(exc_info.value)


class TestValidateHubToken:
    """Tests for validate_hub_token function."""

    def test_provided_token_returned(self):
        """Test that provided token is returned."""
        token = "hf_test_token"
        result = validate_hub_token(token)
        assert result == token

    def test_required_token_raises_when_none(self):
        """Test that required token raises when not available."""
        with patch("unsloth.save.validation.get_token", return_value=None):
            with pytest.raises(ValidationError) as exc_info:
                validate_hub_token(None, required=True)
            assert "required" in str(exc_info.value)
