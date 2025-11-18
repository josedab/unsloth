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
Modular model saving functionality for Unsloth.

This package provides a clean, modular interface for saving models in various formats:
- LoRA adapters
- Merged models (16-bit or 4-bit)
- GGUF format for llama.cpp
- HuggingFace Hub uploads
- Ollama integration

Example usage:
    from unsloth.save import save_merged_model, MergedConfig

    config = MergedConfig(
        save_directory="./my_model",
        save_method="merged_16bit",
        maximum_memory_usage=0.8,
    )
    save_merged_model(model, tokenizer, config)
"""

# Base classes and configs
from .base import (
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

# Validation
from .validation import (
    validate_model,
    validate_tokenizer,
    validate_directory,
    validate_hub_token,
    validate_quantization_method,
    validate_save_method,
    check_sentencepiece_model,
)

# Merged model saving
from .merged import (
    save_merged_model,
    merge_lora_weights,
    fast_save_pickle,
)

# GGUF conversion
from .gguf import (
    save_to_gguf,
    convert_model_to_gguf,
    validate_gguf_config,
)

# HuggingFace Hub
from .hub import (
    push_to_hub,
    create_huggingface_repo,
    upload_to_huggingface,
    determine_username,
    MODEL_CARD,
)

# LoRA saving
from .lora import (
    save_lora_model,
    push_lora_to_hub,
)

# Ollama integration
from .ollama import (
    create_ollama_modelfile,
    create_ollama_model,
    push_to_ollama_hub,
    push_to_ollama,
    fix_tokenizer_bos_token,
)


__all__ = [
    # Configs
    "SaveConfig",
    "GGUFConfig",
    "MergedConfig",
    "LoRAConfig",
    # Exceptions
    "SaveError",
    "ValidationError",
    "ConversionError",
    "HubError",
    # Constants
    "ALLOWED_QUANTS",
    "LLAMA_WEIGHTS",
    "LLAMA_LAYERNORMS",
    "MODEL_CARD",
    # Core functions
    "save_merged_model",
    "save_to_gguf",
    "convert_model_to_gguf",
    "save_lora_model",
    "push_lora_to_hub",
    "push_to_hub",
    # Hub functions
    "create_huggingface_repo",
    "upload_to_huggingface",
    "determine_username",
    # Ollama functions
    "create_ollama_modelfile",
    "create_ollama_model",
    "push_to_ollama_hub",
    "push_to_ollama",
    "fix_tokenizer_bos_token",
    # Validation functions
    "validate_model",
    "validate_tokenizer",
    "validate_directory",
    "validate_hub_token",
    "validate_quantization_method",
    "validate_save_method",
    "validate_gguf_config",
    "check_sentencepiece_model",
    # Utilities
    "merge_lora_weights",
    "print_quantization_methods",
    "fast_save_pickle",
]
