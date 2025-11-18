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

"""GGUF conversion functionality."""

from typing import TYPE_CHECKING, Optional, List, Tuple, Union
from pathlib import Path
import os
import torch
import psutil

from .base import (
    GGUFConfig,
    ConversionError,
    ValidationError,
    ALLOWED_QUANTS,
    IS_KAGGLE_ENVIRONMENT,
)
from .validation import validate_model, validate_quantization_method

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

__all__ = [
    "save_to_gguf",
    "convert_model_to_gguf",
    "validate_gguf_config",
]


def save_to_gguf(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    config: GGUFConfig,
) -> Tuple[List[str], bool, bool]:
    """
    Convert and save model to GGUF format.

    Args:
        model: Model to convert
        tokenizer: Associated tokenizer
        config: GGUF save configuration

    Returns:
        Tuple of (file_locations, want_full_precision, is_vlm)

    Raises:
        ConversionError: GGUF conversion failed
        ValidationError: Invalid configuration
    """
    from unsloth_zoo.hf_utils import dtype_from_config
    from unsloth.models.loader_utils import get_model_name

    # Validate config
    validate_gguf_config(config)

    # Get model info
    try:
        base_model_name = get_model_name(model.config._name_or_path, load_in_4bit=False)
        model_name = base_model_name.split("/")[-1]
    except Exception:
        base_model_name = model.config._name_or_path
        model_name = base_model_name.split("/")[-1]

    # Get model dtype
    model_dtype = _get_model_dtype(model)

    # Check if VLM or GPT-OSS
    is_vlm = _check_is_vlm(model)
    is_gpt_oss = _check_is_gpt_oss(model)

    # Call the core conversion function
    return convert_model_to_gguf(
        model_name=model_name,
        model_type=model.config.model_type,
        model_dtype=model_dtype,
        model_directory=str(config.save_directory),
        quantization_method=config.quantization_method,
        first_conversion=config.first_conversion,
        is_vlm=is_vlm,
        is_gpt_oss=is_gpt_oss,
    )


def convert_model_to_gguf(
    model_name: str,
    model_type: str,
    model_dtype: str,
    model_directory: str,
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: Optional[str] = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
) -> Tuple[List[str], bool, bool]:
    """
    Core GGUF conversion function.

    Args:
        model_name: Name of the model
        model_type: Type of model
        model_dtype: Model data type (float16/bfloat16)
        model_directory: Directory containing model files
        quantization_method: Quantization method(s) to use
        first_conversion: Initial conversion dtype
        is_vlm: Whether model is a vision-language model
        is_gpt_oss: Whether model is GPT-OSS

    Returns:
        Tuple of (file_locations, want_full_precision, is_vlm)

    Raises:
        ConversionError: Conversion failed
    """
    from transformers.models.llama.modeling_llama import logger
    from unsloth_zoo.llama_cpp import (
        convert_to_gguf,
        quantize_gguf,
        use_local_gguf,
        install_llama_cpp,
        check_llama_cpp,
        _download_convert_hf_to_gguf,
    )

    # Print output only if logging enabled
    print_output = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

    # Validate and normalize dtype
    if model_dtype not in ("float16", "bfloat16"):
        raise ValidationError(f"Invalid model_dtype: {model_dtype}")

    model_dtype = "f16" if model_dtype == "float16" else "bf16"

    # Normalize quantization methods
    quantization_method = _normalize_quant_methods(quantization_method, model_dtype)

    # Check bfloat16 support
    if model_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        logger.warning(
            "Unsloth: Cannot convert to bf16 GGUF since your computer doesn't support it.\n"
            "We shall switch instead to f16."
        )
        model_dtype = "f16"

    # Determine first conversion
    first_conversion = _determine_first_conversion(
        first_conversion,
        quantization_method,
        model_dtype,
        is_gpt_oss
    )

    # Print conversion info
    _print_conversion_info(first_conversion, quantization_method)

    # Step 1: Install llama.cpp if needed
    quantizer_location, converter_location = _ensure_llama_cpp(print_output)

    # Step 2-3: Convert to GGUF
    with use_local_gguf():
        converter_path, supported_text_archs, supported_vision_archs = (
            _download_convert_hf_to_gguf()
        )

        first_conversion_dtype = "" if first_conversion == "None" else first_conversion
        print(f"Unsloth: [1] Converting model into {first_conversion_dtype} GGUF format.")
        print("This might take 3 minutes...")

        initial_files, is_vlm_update = convert_to_gguf(
            model_name=model_name,
            input_folder=model_directory,
            model_dtype=model_dtype,
            quantization_type=first_conversion,
            converter_location=converter_path,
            supported_text_archs=supported_text_archs,
            supported_vision_archs=supported_vision_archs,
            is_vlm=is_vlm,
            is_gpt_oss=is_gpt_oss,
            max_shard_size="50GB",
            print_output=print_output,
        )

    # Verify conversion
    _verify_conversion(initial_files)

    print(f"Unsloth: Initial conversion completed! Files: {initial_files}")

    # Step 4: Additional quantizations
    all_saved_locations = _perform_quantizations(
        initial_files,
        quantization_method,
        first_conversion,
        model_name,
        quantizer_location,
        print_output,
        is_gpt_oss
    )

    # Determine if full precision was requested
    if is_gpt_oss:
        want_full_precision = True
    else:
        want_full_precision = first_conversion in frozenset(quantization_method)

    print("Unsloth: All GGUF conversions completed successfully!")
    print(f"Generated files: {all_saved_locations}")

    return all_saved_locations, want_full_precision, is_vlm_update


def validate_gguf_config(config: GGUFConfig) -> None:
    """
    Validate GGUF configuration.

    Args:
        config: GGUF configuration to validate

    Raises:
        ValidationError: Invalid configuration
    """
    for method in config.quantization_method:
        # Validate after normalization
        normalized = method.lower().strip()
        if normalized not in ALLOWED_QUANTS and normalized not in (
            "not_quantized", "fast_quantized", "quantized"
        ):
            valid = ", ".join(ALLOWED_QUANTS.keys())
            raise ValidationError(
                f"Invalid quantization method: {method}\n"
                f"Valid options: {valid}"
            )


def _get_model_dtype(model) -> str:
    """Get model data type as string."""
    from unsloth_zoo.hf_utils import dtype_from_config

    try:
        model_dtype = dtype_from_config(model.config)
        if isinstance(model_dtype, str):
            if model_dtype not in ("float16", "bfloat16"):
                raise TypeError("Invalid dtype")
        elif model_dtype == torch.float16:
            model_dtype = "float16"
        elif model_dtype == torch.bfloat16:
            model_dtype = "bfloat16"
        else:
            raise TypeError("Invalid dtype")
    except Exception as e:
        print(f"Unsloth: Could not determine dtype ({e}), defaulting to float16")
        model_dtype = "float16"

    return model_dtype


def _check_is_vlm(model) -> bool:
    """Check if model is a vision-language model."""
    if hasattr(model, "config") and hasattr(model.config, "architectures"):
        is_vlm = any(
            x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
            for x in model.config.architectures
        )
        return is_vlm or hasattr(model.config, "vision_config")
    return False


def _check_is_gpt_oss(model) -> bool:
    """Check if model is GPT-OSS."""
    if hasattr(model.config, "architectures"):
        if model.config.architectures == "GptOssForCausalLM":
            return True
    if hasattr(model.config, "model_type"):
        if model.config.model_type in ["gpt-oss", "gpt_oss"]:
            return True
    return False


def _normalize_quant_methods(
    quantization_method: Union[str, List[str]],
    model_dtype: str
) -> List[str]:
    """Normalize quantization methods to list format."""
    if isinstance(quantization_method, str):
        quantization_method = [quantization_method]
    elif isinstance(quantization_method, tuple):
        quantization_method = list(quantization_method)

    # Validate and convert methods
    normalized = []
    for method in quantization_method:
        if method.startswith("iq2"):
            raise ConversionError(
                "Currently iq2 type quantizations aren't supported yet"
            )

        # Map aliases
        if method == "not_quantized":
            method = model_dtype
        elif method == "fast_quantized":
            method = "q8_0"
        elif method == "quantized":
            method = "q4_k_m"
        elif method is None:
            method = "q8_0"

        if method not in ALLOWED_QUANTS:
            error = f"Quant method [{method}] not supported. Choose from:\n"
            for key, value in ALLOWED_QUANTS.items():
                error += f"[{key}] => {value}\n"
            raise ValidationError(error)

        normalized.append(method)

    return normalized


def _determine_first_conversion(
    first_conversion: Optional[str],
    quantization_method: List[str],
    model_dtype: str,
    is_gpt_oss: bool
) -> str:
    """Determine optimal first conversion type."""
    if is_gpt_oss:
        return "None"

    if first_conversion is None:
        # Single q8_0 - direct conversion
        if len(quantization_method) == 1 and quantization_method[0] == "q8_0":
            return "None"

        # Determine highest precision needed
        strength = 0
        for method in quantization_method:
            if method == "f32":
                strength = max(strength, 3)
            elif method == "f16":
                strength = max(strength, 2)
            elif method == "bf16":
                strength = max(strength, 1)

        if strength >= 3:
            return "f32"
        elif strength >= 2:
            return "f16"
        elif strength >= 1:
            return "bf16"
        else:
            return "bf16"

    # Check bfloat16 support
    if first_conversion == "bf16" and not torch.cuda.is_bf16_supported():
        return "f16"

    return first_conversion


def _print_conversion_info(first_conversion: str, quantization_method: List[str]):
    """Print GGUF conversion information."""
    first_conversion_dtype = "" if first_conversion == "None" else first_conversion
    print_info = (
        f"==((====))==  Unsloth: Conversion from HF to GGUF information\n"
        f"   {chr(92)}{chr(92)}   /|    [0] Installing llama.cpp might take 3 minutes.\n"
        f"O^O/ {chr(92)}_/ {chr(92)}    [1] Converting HF to GGUF {first_conversion_dtype} might take 3 minutes.\n"
        f"{chr(92)}        /    [2] Converting GGUF {first_conversion_dtype} to {quantization_method} might take 10 minutes each.\n"
        f' "-____-"     In total, you will have to wait at least 16 minutes.\n'
    )
    print(print_info)


def _ensure_llama_cpp(print_output: bool) -> Tuple[str, str]:
    """Ensure llama.cpp is installed."""
    from unsloth_zoo.llama_cpp import check_llama_cpp, install_llama_cpp

    try:
        quantizer_location, converter_location = check_llama_cpp()
        print("Unsloth: llama.cpp found in the system. Skipping installation.")
    except Exception:
        print("Unsloth: Installing llama.cpp. This might take 3 minutes...")
        quantizer_location, converter_location = install_llama_cpp(
            gpu_support=False,
            print_output=print_output,
        )

    return quantizer_location, converter_location


def _verify_conversion(initial_files: List[str]):
    """Verify that conversion produced expected files."""
    for file in initial_files:
        if not os.path.exists(file):
            if IS_KAGGLE_ENVIRONMENT:
                raise ConversionError(
                    f"Conversion failed for {file}\n"
                    "You are in a Kaggle environment with limited disk space (20GB).\n"
                    "Try saving to /tmp for more space or use a smaller model."
                )
            else:
                raise ConversionError(
                    f"Conversion failed for {file}\n"
                    "Please check disk space and try again."
                )


def _perform_quantizations(
    initial_files: List[str],
    quantization_method: List[str],
    first_conversion: str,
    model_name: str,
    quantizer_location: str,
    print_output: bool,
    is_gpt_oss: bool
) -> List[str]:
    """Perform additional quantizations."""
    from unsloth_zoo.llama_cpp import quantize_gguf

    all_saved_locations = initial_files.copy()

    if is_gpt_oss:
        print("Unsloth: GPT-OSS model - skipping additional quantizations")
        return all_saved_locations

    base_gguf = initial_files[0]
    quants_created = False
    first_conversion_dtype = "" if first_conversion == "None" else first_conversion

    for quant_method in quantization_method:
        if quant_method != first_conversion:
            print(
                f"Unsloth: [2] Converting GGUF {first_conversion_dtype} into {quant_method}. "
                "This might take 10 minutes..."
            )
            output_location = f"{model_name}.{quant_method.upper()}.gguf"

            try:
                quantized_file = quantize_gguf(
                    input_gguf=base_gguf,
                    output_gguf=output_location,
                    quant_type=quant_method,
                    quantizer_location=quantizer_location,
                    print_output=print_output,
                )
                all_saved_locations.append(quantized_file)
                quants_created = True
            except Exception as e:
                if IS_KAGGLE_ENVIRONMENT:
                    raise ConversionError(
                        f"Quantization failed for {output_location}\n"
                        "Kaggle only provides 20GB of disk space.\n"
                        f"Error: {e}"
                    )
                else:
                    raise ConversionError(
                        f"Quantization failed for {output_location}\n"
                        "You might need to compile llama.cpp yourself.\n"
                        f"Error: {e}"
                    )

    # Cleanup base file if quants were created
    print("Unsloth: Model files cleanup...")
    if quants_created:
        all_saved_locations.remove(base_gguf)
        Path(base_gguf).unlink()
        all_saved_locations.reverse()

    return all_saved_locations
