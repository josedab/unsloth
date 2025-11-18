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

"""Tests to verify all imports work correctly."""

import pytest


class TestModuleImports:
    """Tests for module imports."""

    def test_base_imports(self):
        """Test that base module imports work."""
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
            LLAMA_WEIGHTS,
            LLAMA_LAYERNORMS,
            print_quantization_methods,
        )
        assert SaveConfig is not None
        assert ALLOWED_QUANTS is not None

    def test_validation_imports(self):
        """Test that validation module imports work."""
        from unsloth.save.validation import (
            validate_model,
            validate_tokenizer,
            validate_directory,
            validate_hub_token,
            validate_quantization_method,
            validate_save_method,
            check_sentencepiece_model,
        )
        assert validate_model is not None

    def test_merged_imports(self):
        """Test that merged module imports work."""
        from unsloth.save.merged import (
            save_merged_model,
            merge_lora_weights,
            fast_save_pickle,
        )
        assert save_merged_model is not None

    def test_gguf_imports(self):
        """Test that gguf module imports work."""
        from unsloth.save.gguf import (
            save_to_gguf,
            convert_model_to_gguf,
            validate_gguf_config,
        )
        assert save_to_gguf is not None

    def test_hub_imports(self):
        """Test that hub module imports work."""
        from unsloth.save.hub import (
            push_to_hub,
            create_huggingface_repo,
            upload_to_huggingface,
            determine_username,
            MODEL_CARD,
        )
        assert push_to_hub is not None
        assert MODEL_CARD is not None

    def test_lora_imports(self):
        """Test that lora module imports work."""
        from unsloth.save.lora import (
            save_lora_model,
            push_lora_to_hub,
        )
        assert save_lora_model is not None

    def test_ollama_imports(self):
        """Test that ollama module imports work."""
        from unsloth.save.ollama import (
            create_ollama_modelfile,
            create_ollama_model,
            push_to_ollama_hub,
            push_to_ollama,
            fix_tokenizer_bos_token,
        )
        assert create_ollama_modelfile is not None

    def test_package_imports(self):
        """Test that package-level imports work."""
        from unsloth.save import (
            SaveConfig,
            GGUFConfig,
            MergedConfig,
            LoRAConfig,
            SaveError,
            ValidationError,
            ConversionError,
            HubError,
            save_merged_model,
            save_to_gguf,
            save_lora_model,
            push_to_hub,
            create_ollama_modelfile,
        )
        assert SaveConfig is not None
        assert save_merged_model is not None

    def test_backward_compatibility_imports(self):
        """Test that backward compatibility imports from unsloth.save work."""
        # These are now imported into the original save.py for backward compatibility
        from unsloth.save import (
            SaveConfig,
            GGUFConfig,
            MergedConfig,
            LoRAConfig,
            SaveError,
            ValidationError,
            ConversionError,
            HubError,
            print_quantization_methods,
            create_ollama_modelfile,
        )
        assert SaveConfig is not None


class TestAllExports:
    """Test that __all__ exports are correct."""

    def test_save_package_all(self):
        """Test that unsloth.save exports all expected items."""
        import unsloth.save as save_module

        expected = [
            "SaveConfig",
            "GGUFConfig",
            "MergedConfig",
            "LoRAConfig",
            "SaveError",
            "ValidationError",
            "ConversionError",
            "HubError",
            "save_merged_model",
            "save_to_gguf",
            "save_lora_model",
        ]

        for item in expected:
            assert item in save_module.__all__, f"{item} not in __all__"
            assert hasattr(save_module, item), f"{item} not accessible"
