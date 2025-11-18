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

"""Merged model saving functionality."""

from typing import TYPE_CHECKING, Optional, Tuple, Union
from pathlib import Path
import os
import gc
import pickle
import torch
import psutil
import re
import shutil

from .base import (
    MergedConfig,
    SaveError,
    ValidationError,
    LLAMA_WEIGHTS,
    LLAMA_LAYERNORMS,
    IS_COLAB_ENVIRONMENT,
    IS_KAGGLE_ENVIRONMENT,
    KAGGLE_TMP,
)
from .validation import validate_model, validate_tokenizer, validate_directory

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "save_merged_model",
    "merge_lora_weights",
]


def save_merged_model(
    model: "PreTrainedModel",
    tokenizer: Optional["PreTrainedTokenizerBase"],
    config: MergedConfig,
) -> Tuple[Path, Optional[str]]:
    """
    Save model with LoRA weights merged into base model.

    Args:
        model: Trained model (PEFT or base)
        tokenizer: Associated tokenizer
        config: Merge save configuration

    Returns:
        Tuple of (save_directory, username) where username is set if pushing to hub

    Raises:
        ValidationError: Invalid model or config
        SaveError: Save operation failed
    """
    from transformers.models.llama.modeling_llama import logger
    from unsloth_zoo.hf_utils import dtype_from_config

    # Validate inputs
    validate_model(model)
    validate_tokenizer(tokenizer)
    save_directory = validate_directory(config.save_directory)

    # Get internal model
    from peft import PeftModelForCausalLM
    if isinstance(model, PeftModelForCausalLM):
        internal_model = model.model
    else:
        internal_model = model

    # Clean memory
    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()

    print("Unsloth: Merging 4bit and LoRA weights to 16bit...")

    # Calculate memory usage
    max_ram = _calculate_max_ram(config)

    # Handle Kaggle environment
    temporary_location = config.temporary_location
    if IS_KAGGLE_ENVIRONMENT:
        temporary_location = os.path.join(KAGGLE_TMP, temporary_location)

    # Create temporary directory
    if not os.path.exists(temporary_location):
        os.makedirs(temporary_location)

    # Free cached model in constrained environments
    if IS_KAGGLE_ENVIRONMENT or IS_COLAB_ENVIRONMENT:
        logger.warning_once(
            "Unsloth: Kaggle/Colab has limited disk space. We need to delete the downloaded\n"
            "model which will save 4-16GB of disk space."
        )
        _free_cached_model(internal_model)

    # Build state dict with merged weights
    state_dict = _build_merged_state_dict(
        internal_model,
        max_ram,
        config.maximum_memory_usage,
        temporary_location
    )

    # Save tokenizer
    username = _save_tokenizer(tokenizer, save_directory, config)

    # Update model config (remove quantization_config for merged model)
    old_config = model.config
    new_config = _create_merged_config(model)

    # Save the model
    _save_merged_weights(
        model,
        internal_model,
        state_dict,
        save_directory,
        config,
        new_config
    )

    # Restore original config
    _restore_config(model, old_config)

    # Cleanup
    _cleanup_state_dict(state_dict)
    shutil.rmtree(temporary_location, ignore_errors=True)

    for _ in range(3):
        torch.cuda.empty_cache()
        gc.collect()

    print("Done.")
    return save_directory, username


def merge_lora_weights(layer, name: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Merge LoRA weights into base layer weights.

    Args:
        layer: Layer to merge
        name: Layer name for error reporting

    Returns:
        Tuple of (merged_weights, bias)

    Raises:
        ValueError: If merge results in non-finite values
    """
    from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
    from peft.tuners.lora import Linear4bit as Peft_Linear4bit
    from peft.tuners.lora import Linear as Peft_Linear
    from unsloth.kernels import fast_dequantize, QUANT_STATE, get_lora_parameters_bias

    bias = getattr(layer, "bias", None)

    if isinstance(layer, (Bnb_Linear4bit, Peft_Linear4bit, Peft_Linear)):
        # Is LoRA so we need to merge
        W, quant_state, A, B, s, bias = get_lora_parameters_bias(layer)

        if quant_state is not None:
            dtype = (
                quant_state.dtype if type(quant_state) is not list else quant_state[2]
            )
            W = fast_dequantize(W, quant_state)
        else:
            dtype = W.dtype

        W = W.to(torch.float32).t()

        if A is not None:
            W.addmm_(A.t().to(torch.float32), B.t().to(torch.float32), alpha=s)

            maximum_element = torch.max(W.min().abs(), W.max())
            if not torch.isfinite(maximum_element).item():
                raise ValueError(
                    f"Unsloth: Merge failed.\n{name} has some elements = infinity."
                )

        W = W.t().to(dtype)
    else:
        W = layer.weight

    return W, bias


def _calculate_max_ram(config: MergedConfig) -> int:
    """Calculate maximum RAM available for saving."""
    max_ram = psutil.virtual_memory().available
    sharded_ram_usage = 5 * 1024 * 1024 * 1024  # 5GB default

    # Parse max_shard_size
    if isinstance(config.max_shard_size, str):
        gb_found = re.match(
            r"([0-9]{1,})[\s]{0,}GB", config.max_shard_size, flags=re.IGNORECASE
        )
        mb_found = re.match(
            r"([0-9]{1,})[\s]{0,}MB", config.max_shard_size, flags=re.IGNORECASE
        )
        if gb_found:
            sharded_ram_usage = int(gb_found.group(1)) * 1024 * 1024 * 1024
        elif mb_found:
            sharded_ram_usage = int(mb_found.group(1)) * 1024 * 1024
    elif isinstance(config.max_shard_size, int):
        sharded_ram_usage = config.max_shard_size

    # Adjust based on serialization type
    if config.safe_serialization:
        max_ram -= sharded_ram_usage
    else:
        max_ram -= sharded_ram_usage * 0.25

    max_ram = int(max(0, max_ram) * config.maximum_memory_usage)

    print(
        f"Unsloth: Will use up to "
        f"{round(max_ram / 1024 / 1024 / 1024, 2)} out of "
        f"{round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)} RAM for saving."
    )

    return max_ram


def _free_cached_model(internal_model):
    """Free cached model to save disk space."""
    from huggingface_hub import scan_cache_dir
    from transformers.models.llama.modeling_llama import logger

    cached_repos = list(scan_cache_dir().repos)

    for cached_repo in cached_repos:
        if cached_repo.repo_id == internal_model.config._name_or_path:
            remove_cache_commit = list(cached_repo.revisions)[0].commit_hash
            delete_strategy = scan_cache_dir().delete_revisions(remove_cache_commit)

            logger.warning_once(
                "Unsloth: Will remove a cached repo with size "
                + delete_strategy.expected_freed_size_str
            )

            delete_strategy.execute()


def _build_merged_state_dict(
    internal_model,
    max_ram: int,
    maximum_memory_usage: float,
    temporary_location: str
) -> dict:
    """Build state dict with merged LoRA weights."""
    from collections import OrderedDict
    from transformers.models.llama.modeling_llama import logger
    from unsloth_zoo.hf_utils import dtype_from_config
    from tqdm import tqdm as ProgressBar

    state_dict = OrderedDict()

    # Get torch dtype
    torch_dtype = dtype_from_config(internal_model.config)
    if isinstance(torch_dtype, str):
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16

    # Save embed tokens
    state_dict["model.embed_tokens.weight"] = (
        internal_model.model.embed_tokens.weight.data.to(torch_dtype)
    )

    max_vram = int(
        torch.cuda.get_device_properties(0).total_memory * maximum_memory_usage
    )

    print("Unsloth: Saving model... This might take 5 minutes ...")

    # Process each layer
    for j, layer in enumerate(ProgressBar(internal_model.model.layers)):
        for item in LLAMA_WEIGHTS:
            proj = eval(f"layer.{item}")
            name = f"model.layers.{j}.{item}.weight"
            W, bias = merge_lora_weights(proj, name)

            # Save bias if present
            if bias is not None:
                state_dict[f"model.layers.{j}.{item}.bias"] = bias

            # Decide where to save based on memory
            if (torch.cuda.memory_allocated() + W.nbytes) < max_vram:
                state_dict[name] = W
            else:
                # Save to disk
                logger.warning_once("\nWe will save to Disk and not RAM now.")
                filename = os.path.join(temporary_location, f"{name}.pt")
                torch.save(
                    W,
                    filename,
                    pickle_module=pickle,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )
                state_dict[name] = torch.load(
                    filename, map_location="cpu", mmap=True, weights_only=False
                )

        # Save layer norms
        for item in LLAMA_LAYERNORMS:
            try:
                state_dict[f"model.layers.{j}.{item}.weight"] = eval(
                    f"layer.{item}.weight.data"
                )
            except Exception:
                continue

    # Save final norm and lm_head
    state_dict["model.norm.weight"] = internal_model.model.norm.weight.data

    # Check for tied weights
    if (
        internal_model.model.embed_tokens.weight.data_ptr()
        != internal_model.lm_head.weight.data_ptr()
    ):
        state_dict["lm_head.weight"] = internal_model.lm_head.weight.data.to(torch_dtype)

    # Ensure all values are tensors
    for key, value in state_dict.items():
        if hasattr(value, "data"):
            state_dict[key] = value = value.data
        if not isinstance(value, torch.Tensor):
            logger.warning_once(
                f"Unsloth: {key} is not a Tensor but a {type(value)}."
            )

    return state_dict


def _save_tokenizer(
    tokenizer: Optional["PreTrainedTokenizerBase"],
    save_directory: Path,
    config: MergedConfig
) -> Optional[str]:
    """Save tokenizer with proper settings."""
    if tokenizer is None:
        print()
        return None

    print("Unsloth: Saving tokenizer...", end="")

    # Set padding side to left for inference
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    tokenizer.save_pretrained(str(save_directory))

    # Restore padding side
    tokenizer.padding_side = old_padding_side

    print(" Done.")
    return None


def _create_merged_config(model):
    """Create config without quantization settings for merged model."""
    new_config = model.config.to_dict()
    if "quantization_config" in new_config:
        del new_config["quantization_config"]
    return type(model.config).from_dict(new_config)


def _save_merged_weights(
    model,
    internal_model,
    state_dict: dict,
    save_directory: Path,
    config: MergedConfig,
    new_config
):
    """Save the merged model weights."""
    # Update model config
    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = new_config
    model.config = new_config

    # Add model tags
    if hasattr(model, "add_model_tags"):
        model.add_model_tags(["unsloth"])

    # Prepare save settings
    save_settings = {
        "save_directory": str(save_directory),
        "state_dict": state_dict,
        "safe_serialization": config.safe_serialization,
        "max_shard_size": config.max_shard_size,
    }

    internal_model.save_pretrained(**save_settings)


def _restore_config(model, old_config):
    """Restore original model config."""
    original_model = model
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = old_config
    model.config = old_config


def _cleanup_state_dict(state_dict: dict):
    """Clean up state dict to free memory."""
    for j, (key, value) in enumerate(state_dict.items()):
        state_dict[key] = None
        if j % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()


def fast_save_pickle(shard, name: str):
    """Fast pickle-based saving for slow CPUs."""
    print(f"Unsloth: Saving {name}...")
    torch.save(shard, name)
