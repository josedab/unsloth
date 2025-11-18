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

"""Input validation for model saving operations."""

from typing import TYPE_CHECKING, Optional, Union
from pathlib import Path
import os

from .base import ValidationError, ALLOWED_QUANTS

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "validate_model",
    "validate_tokenizer",
    "validate_directory",
    "validate_hub_token",
    "validate_quantization_method",
    "validate_save_method",
]


def validate_model(model: "PreTrainedModel") -> None:
    """
    Validate that a model is suitable for saving.

    Args:
        model: Model to validate

    Raises:
        ValidationError: If model is invalid
    """
    if model is None:
        raise ValidationError("Model cannot be None")

    if not hasattr(model, "config"):
        raise ValidationError("Model must have a 'config' attribute")

    # Check for basic model attributes
    if not hasattr(model, "state_dict"):
        raise ValidationError("Model must have a 'state_dict' method")


def validate_tokenizer(
    tokenizer: Optional["PreTrainedTokenizerBase"],
    require: bool = False
) -> None:
    """
    Validate tokenizer for saving.

    Args:
        tokenizer: Tokenizer to validate
        require: If True, raise error when tokenizer is None

    Raises:
        ValidationError: If tokenizer is invalid
    """
    if tokenizer is None:
        if require:
            raise ValidationError("Tokenizer is required for this operation")
        return

    if not hasattr(tokenizer, "save_pretrained"):
        raise ValidationError("Tokenizer must have a 'save_pretrained' method")


def validate_directory(
    directory: Union[str, Path],
    create: bool = True
) -> Path:
    """
    Validate and optionally create a save directory.

    Args:
        directory: Directory path
        create: If True, create directory if it doesn't exist

    Returns:
        Path object for the directory

    Raises:
        ValidationError: If directory is invalid or cannot be created
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValidationError(
                f"Cannot create directory '{directory}': {e}"
            ) from e
    elif not directory.exists():
        raise ValidationError(f"Directory does not exist: {directory}")

    if directory.exists() and not directory.is_dir():
        raise ValidationError(f"Path exists but is not a directory: {directory}")

    return directory


def validate_hub_token(token: Optional[str], required: bool = False) -> Optional[str]:
    """
    Validate HuggingFace Hub token.

    Args:
        token: HuggingFace token
        required: If True, token must be provided

    Returns:
        The validated token or retrieved token

    Raises:
        ValidationError: If token is required but not available
    """
    if token is not None:
        return token

    # Try to get token from huggingface_hub
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        try:
            from huggingface_hub.utils import get_token
            token = get_token()
        except ImportError:
            from huggingface_hub.utils._token import get_token
            token = get_token()

    if required and token is None:
        raise ValidationError(
            "HuggingFace Hub token is required.\n"
            "Please provide a token or login with `huggingface-cli login`.\n"
            "Get your token at https://huggingface.co/settings/tokens"
        )

    return token


def validate_quantization_method(method: str) -> str:
    """
    Validate a quantization method.

    Args:
        method: Quantization method name

    Returns:
        Normalized quantization method name

    Raises:
        ValidationError: If method is not supported
    """
    method = method.lower().strip()

    # Handle aliases
    if method == "not_quantized":
        return method
    elif method == "fast_quantized":
        return method
    elif method == "quantized":
        return method

    if method not in ALLOWED_QUANTS:
        valid_methods = ", ".join(ALLOWED_QUANTS.keys())
        raise ValidationError(
            f"Invalid quantization method: '{method}'\n"
            f"Valid methods: {valid_methods}"
        )

    return method


def validate_save_method(method: str) -> str:
    """
    Validate a save method.

    Args:
        method: Save method name

    Returns:
        Normalized save method name

    Raises:
        ValidationError: If method is not valid
    """
    method = method.lower().replace(" ", "_")

    valid_methods = ["lora", "merged_16bit", "merged_4bit", "merged_4bit_forced"]

    if method not in valid_methods:
        raise ValidationError(
            f"Invalid save method: '{method}'\n"
            f"Valid methods:\n"
            f'  "lora"         - Saves LoRA modules only (fastest)\n'
            f'  "merged_16bit" - Merges LoRA weights to float16 (for GGUF/llama.cpp)\n'
            f'  "merged_4bit"  - Merges LoRA weights to 4bit (for DPO/inference)'
        )

    # Convert forced variant
    if method == "merged_4bit_forced":
        return "merged_4bit"

    return method


def check_sentencepiece_model(
    model: "PreTrainedModel",
    temporary_location: str = "_unsloth_sentencepiece_temp"
) -> bool:
    """
    Check if a model uses a SentencePiece tokenizer.

    Args:
        model: Model to check
        temporary_location: Temporary directory for checking

    Returns:
        True if model uses SentencePiece tokenizer
    """
    import shutil

    if not hasattr(model, "_saved_temp_tokenizer"):
        return False

    temp_tokenizer = model._saved_temp_tokenizer
    sentencepiece_model = False
    file_location = os.path.join(temporary_location, temp_tokenizer.name_or_path)
    created_folder = False

    if not os.path.exists(file_location):
        created_folder = True
        os.makedirs(file_location)

    temp_tokenizer.save_pretrained(file_location)

    if os.path.isfile(f"{file_location}/tokenizer.model"):
        sentencepiece_model = True

    if created_folder:
        shutil.rmtree(file_location, ignore_errors=True)

    return sentencepiece_model
