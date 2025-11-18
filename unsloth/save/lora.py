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

"""LoRA adapter saving functionality."""

from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path
import gc

from .base import LoRAConfig, SaveError
from .validation import validate_model, validate_tokenizer, validate_directory
from .hub import upload_to_huggingface, determine_username

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "save_lora_model",
    "push_lora_to_hub",
]


def save_lora_model(
    model: "PreTrainedModel",
    tokenizer: Optional["PreTrainedTokenizerBase"],
    config: LoRAConfig,
) -> Tuple[Path, Optional[str]]:
    """
    Save LoRA adapters without merging.

    Args:
        model: Model with LoRA adapters
        tokenizer: Associated tokenizer
        config: LoRA save configuration

    Returns:
        Tuple of (save_directory, username)

    Raises:
        SaveError: Save operation failed
    """
    # Validate inputs
    validate_model(model)
    validate_tokenizer(tokenizer)
    save_directory = validate_directory(config.save_directory)

    print("Unsloth: Saving LoRA adapters. Please wait...")

    # Add model tags
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth"])

    # Save tokenizer first
    if tokenizer is not None:
        print("Unsloth: Saving tokenizer...", end="")
        _save_tokenizer(tokenizer, save_directory)
        print(" Done.")

    # Save model
    print("Unsloth: Saving model...", end="")

    save_settings = {
        "save_directory": str(save_directory),
        "safe_serialization": config.safe_serialization,
        "max_shard_size": config.max_shard_size,
        "selected_adapters": config.selected_adapters,
    }

    model.save_pretrained(**save_settings)
    print(" Done.")

    # Clean up
    for _ in range(3):
        gc.collect()

    return save_directory, None


def push_lora_to_hub(
    model: "PreTrainedModel",
    tokenizer: Optional["PreTrainedTokenizerBase"],
    config: LoRAConfig,
) -> str:
    """
    Push LoRA adapters to HuggingFace Hub.

    Args:
        model: Model with LoRA adapters
        tokenizer: Associated tokenizer
        config: LoRA save configuration

    Returns:
        Hub URL

    Raises:
        SaveError: Push operation failed
    """
    from .validation import validate_hub_token

    # Validate inputs
    validate_model(model)
    validate_tokenizer(tokenizer)

    token = validate_hub_token(config.token, required=True)
    repo_id = config.repo_id or str(config.save_directory)

    print("Unsloth: Saving LoRA adapters. Please wait...")

    # Add model tags
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth"])

    # Upload metadata
    upload_to_huggingface(
        model=model,
        save_directory=repo_id,
        token=token,
        method="finetuned",
        extra="trl",
        file_location=None,
        old_username=None,
        private=config.private,
    )

    # Push model
    getattr(model, "original_push_to_hub", model.push_to_hub)(
        repo_id=repo_id,
        token=token,
        private=config.private,
        max_shard_size=config.max_shard_size,
        safe_serialization=config.safe_serialization,
        commit_message=config.commit_message,
        tags=config.tags,
    )

    # Push tokenizer
    if tokenizer is not None:
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        getattr(tokenizer, "original_push_to_hub", tokenizer.push_to_hub)(
            repo_id=repo_id,
            token=token,
            private=config.private,
            max_shard_size=config.max_shard_size,
            safe_serialization=config.safe_serialization,
            commit_message=config.commit_message,
            tags=config.tags,
        )

        tokenizer.padding_side = old_padding_side

    # Get full URL
    if "/" not in repo_id:
        from huggingface_hub import whoami
        username = whoami(token=token)["name"]
        hub_url = f"https://huggingface.co/{username}/{repo_id}"
    else:
        hub_url = f"https://huggingface.co/{repo_id}"

    print(f"Saved LoRA model to {hub_url}")

    # Clean up
    for _ in range(3):
        gc.collect()

    return hub_url


def _save_tokenizer(
    tokenizer: "PreTrainedTokenizerBase",
    save_directory: Path
) -> None:
    """Save tokenizer with proper settings."""
    # Set padding side to left for inference
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    tokenizer_save_settings = {
        "save_directory": str(save_directory),
        "legacy_format": None,
        "filename_prefix": None,
        "push_to_hub": False,
        "private": None,
        "token": None,
    }

    tokenizer.save_pretrained(**tokenizer_save_settings)

    # Restore padding side
    tokenizer.padding_side = old_padding_side
