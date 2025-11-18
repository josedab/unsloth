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
Type definitions for the Unsloth library.

This module provides shared type aliases and type definitions used throughout
the Unsloth public API for better IDE support and static analysis.
"""

from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    Any,
    Literal,
    Callable,
    TYPE_CHECKING,
)
from pathlib import Path
import os

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
    from peft import PeftModel
    from datasets import Dataset, DatasetDict

__all__ = [
    # Path types
    "ModelName",
    "PathLike",
    # Data types
    "DType",
    "TorchDType",
    # Device mapping
    "DeviceMap",
    # Quantization
    "QuantizationMethod",
    "SaveMethod",
    # LoRA configuration
    "BiasType",
    "GradientCheckpointing",
    # Return types
    "ModelTokenizerTuple",
    # Chat template types
    "ChatTemplateType",
    "MappingType",
]

# Path and model name types
ModelName = Union[str, Path]
PathLike = Union[str, os.PathLike]

# Data type configurations
DType = Optional[Literal["float16", "bfloat16", "float32"]]
TorchDType = Optional["torch.dtype"]

# Device mapping configuration
DeviceMap = Union[str, Dict[str, Union[int, str]]]

# GGUF quantization methods
QuantizationMethod = Literal[
    "q2_k",
    "q3_k_m",
    "q4_0",
    "q4_1",
    "q4_k_m",
    "q4_k_s",
    "q5_0",
    "q5_1",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q8_0",
    "f16",
    "f32",
    "bf16",
    "not_quantized",
    "fast_quantized",
    "quantized",
]

# Save method options
SaveMethod = Literal[
    "merged_16bit",
    "merged_4bit",
    "lora",
    "16bit",
    "4bit",
]

# LoRA bias configuration
BiasType = Literal["none", "all", "lora_only"]

# Gradient checkpointing options
GradientCheckpointing = Union[bool, Literal["unsloth"]]

# Return types
ModelTokenizerTuple = Tuple["PreTrainedModel", "PreTrainedTokenizer"]

# Chat template types
ChatTemplateType = Optional[Union[str, Literal[
    "unsloth",
    "zephyr",
    "chatml",
    "mistral",
    "llama-2",
    "llama-3",
    "alpaca",
    "vicuna",
    "vicuna_old",
    "phi-3",
    "gemma",
    "gemma_chatml",
    "qwen-2",
    "qwen-2.5",
]]]

# Mapping type for chat templates
MappingType = Optional[Dict[str, str]]

# Token type for HuggingFace Hub authentication
HubToken = Optional[Union[str, bool]]

# Dataset types
DatasetType = Optional[Union["Dataset", "DatasetDict"]]
EvalDatasetType = Optional[Union["Dataset", Dict[str, "Dataset"]]]

# Tokenizer types
TokenizerType = Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]
