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

"""Tests for unsloth.save.base module."""

import pytest
from pathlib import Path
import tempfile

from unsloth.save.base import (
    SaveConfig,
    GGUFConfig,
    MergedConfig,
    LoRAConfig,
    SaveError,
    ValidationError,
    ConversionError,
    HubError,
    ALLOWED_QUANTS,
    print_quantization_methods,
)


class TestSaveConfig:
    """Tests for SaveConfig dataclass."""

    def test_basic_creation(self):
        """Test basic SaveConfig creation."""
        config = SaveConfig(save_directory="/tmp/test")
        assert config.save_directory == Path("/tmp/test")
        assert config.push_to_hub is False
        assert config.private is False
        assert "unsloth" in config.tags

    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = SaveConfig(save_directory="./my_model")
        assert isinstance(config.save_directory, Path)

    def test_tags_include_unsloth(self):
        """Test that tags always include 'unsloth'."""
        config = SaveConfig(save_directory="/tmp", tags=["custom"])
        assert "unsloth" in config.tags
        assert "custom" in config.tags

    def test_commit_message_unsloth_append(self):
        """Test that commit message includes Unsloth branding."""
        config = SaveConfig(
            save_directory="/tmp",
            commit_message="My model"
        )
        assert "Unsloth" in config.commit_message


class TestGGUFConfig:
    """Tests for GGUFConfig dataclass."""

    def test_default_quantization(self):
        """Test default quantization method."""
        config = GGUFConfig(save_directory="/tmp")
        assert config.quantization_method == ["fast_quantized"]

    def test_string_to_list_conversion(self):
        """Test that string quantization method is converted to list."""
        config = GGUFConfig(
            save_directory="/tmp",
            quantization_method="q4_k_m"
        )
        assert isinstance(config.quantization_method, list)
        assert config.quantization_method == ["q4_k_m"]

    def test_multiple_methods(self):
        """Test multiple quantization methods."""
        config = GGUFConfig(
            save_directory="/tmp",
            quantization_method=["q4_k_m", "q8_0"]
        )
        assert len(config.quantization_method) == 2


class TestMergedConfig:
    """Tests for MergedConfig dataclass."""

    def test_default_values(self):
        """Test default MergedConfig values."""
        config = MergedConfig(save_directory="/tmp")
        assert config.save_method == "merged_16bit"
        assert config.maximum_memory_usage == 0.75

    def test_invalid_memory_usage(self):
        """Test that invalid memory usage raises error."""
        with pytest.raises(ValidationError):
            MergedConfig(save_directory="/tmp", maximum_memory_usage=1.5)

        with pytest.raises(ValidationError):
            MergedConfig(save_directory="/tmp", maximum_memory_usage=0)


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Test default LoRAConfig values."""
        config = LoRAConfig(save_directory="/tmp")
        assert config.save_peft_format is True
        assert config.selected_adapters is None


class TestExceptions:
    """Tests for custom exceptions."""

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from SaveError."""
        assert issubclass(ValidationError, SaveError)
        assert issubclass(ConversionError, SaveError)
        assert issubclass(HubError, SaveError)

    def test_exception_messages(self):
        """Test that exceptions can be raised with messages."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid input")
        assert "Invalid input" in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""

    def test_allowed_quants(self):
        """Test ALLOWED_QUANTS contains expected methods."""
        assert "q4_k_m" in ALLOWED_QUANTS
        assert "q8_0" in ALLOWED_QUANTS
        assert "f16" in ALLOWED_QUANTS
        assert "fast_quantized" in ALLOWED_QUANTS

    def test_print_quantization_methods(self, capsys):
        """Test print_quantization_methods output."""
        print_quantization_methods()
        captured = capsys.readouterr()
        assert "q4_k_m" in captured.out
        assert "Recommended" in captured.out
