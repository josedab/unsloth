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

"""Base classes and configurations for model saving."""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Union
from pathlib import Path
import os

__all__ = [
    "SaveConfig",
    "GGUFConfig",
    "MergedConfig",
    "LoRAConfig",
    "SaveError",
    "ValidationError",
    "ConversionError",
    "HubError",
    "ALLOWED_QUANTS",
    "LLAMA_WEIGHTS",
    "LLAMA_LAYERNORMS",
]


# Environment checks
keynames = "\n" + "\n".join(os.environ.keys())
IS_COLAB_ENVIRONMENT = "\nCOLAB_" in keynames
IS_KAGGLE_ENVIRONMENT = "\nKAGGLE_" in keynames
KAGGLE_TMP = "/tmp"
del keynames


# Weights to merge
LLAMA_WEIGHTS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

LLAMA_LAYERNORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


# Quantization methods
ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "f16": "Float16  - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k": "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l": "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m": "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "q3_k_xs": "3-bit extra small quantization",
}


# Exception classes
class SaveError(Exception):
    """Base exception for save operations."""
    pass


class ValidationError(SaveError):
    """Invalid configuration or input."""
    pass


class ConversionError(SaveError):
    """Model conversion failed."""
    pass


class HubError(SaveError):
    """HuggingFace Hub operation failed."""
    pass


# Configuration dataclasses
@dataclass
class SaveConfig:
    """Base configuration for model saving."""
    save_directory: Union[str, Path]
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    token: Optional[str] = None
    private: bool = False
    max_shard_size: Union[int, str] = "5GB"
    safe_serialization: bool = True
    commit_message: Optional[str] = "Trained with Unsloth"
    commit_description: str = "Upload model trained with Unsloth 2x faster"
    tags: Optional[List[str]] = None

    def __post_init__(self):
        """Convert save_directory to Path and set default tags."""
        if isinstance(self.save_directory, str):
            self.save_directory = Path(self.save_directory)

        if self.tags is None:
            self.tags = ["unsloth"]
        elif "unsloth" not in self.tags:
            self.tags = list(self.tags) + ["unsloth"]

        # Ensure commit message includes Unsloth
        if self.commit_message and "Unsloth" not in self.commit_message:
            self.commit_message += " (Trained with Unsloth)"

        if self.commit_description and "Unsloth 2x faster" not in self.commit_description:
            self.commit_description += " (Trained with Unsloth 2x faster)"


@dataclass
class GGUFConfig(SaveConfig):
    """Configuration for GGUF export."""
    quantization_method: Union[str, List[str]] = "fast_quantized"
    first_conversion: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize quantization methods."""
        super().__post_init__()

        # Convert to list if string
        if isinstance(self.quantization_method, str):
            self.quantization_method = [self.quantization_method]
        elif isinstance(self.quantization_method, tuple):
            self.quantization_method = list(self.quantization_method)


@dataclass
class MergedConfig(SaveConfig):
    """Configuration for merged model saving."""
    save_method: Literal["merged_16bit", "merged_4bit"] = "merged_16bit"
    maximum_memory_usage: float = 0.75
    temporary_location: str = "_unsloth_temporary_saved_buffers"

    def __post_init__(self):
        """Validate merged config."""
        super().__post_init__()

        if self.maximum_memory_usage <= 0 or self.maximum_memory_usage > 0.95:
            raise ValidationError(
                f"maximum_memory_usage must be between 0 and 0.95, got {self.maximum_memory_usage}"
            )


@dataclass
class LoRAConfig(SaveConfig):
    """Configuration for LoRA adapter saving."""
    save_peft_format: bool = True
    selected_adapters: Optional[List[str]] = None


def print_quantization_methods():
    """Print all available quantization methods with descriptions."""
    for key, value in ALLOWED_QUANTS.items():
        print(f'"{key}"  ==> {value}')
