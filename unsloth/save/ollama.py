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

"""Ollama integration for model deployment."""

from typing import TYPE_CHECKING, Optional
import subprocess
import sys
import re

from .base import SaveError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "create_ollama_modelfile",
    "create_ollama_model",
    "push_to_ollama_hub",
    "push_to_ollama",
]


def create_ollama_modelfile(
    tokenizer: "PreTrainedTokenizerBase",
    base_model_name: str,
    model_location: str
) -> Optional[str]:
    """
    Create an Ollama Modelfile.

    Args:
        tokenizer: Tokenizer for template
        base_model_name: Base model name for template lookup
        model_location: Path to GGUF file

    Returns:
        Modelfile content or None if no template found
    """
    from unsloth.ollama_template_mappers import (
        OLLAMA_TEMPLATES,
        MODEL_TO_OLLAMA_TEMPLATE_MAPPER
    )

    # Get template name
    ollama_template_name = MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(base_model_name)
    if not ollama_template_name:
        print(
            f"Unsloth: No Ollama template mapping found for model '{base_model_name}'. "
            "Skipping Ollama Modelfile"
        )
        return None

    # Get template
    ollama_modelfile = OLLAMA_TEMPLATES.get(ollama_template_name)
    if not ollama_modelfile:
        print(
            f"Unsloth: No Ollama template found for '{ollama_template_name}'. "
            "Skipping Ollama Modelfile"
        )
        return None

    # Store on tokenizer for reference
    tokenizer._ollama_modelfile = ollama_modelfile
    modelfile = ollama_modelfile

    # Process template with special character handling
    modelfile = _process_modelfile_template(modelfile, model_location, tokenizer)

    return modelfile


def _process_modelfile_template(
    modelfile: str,
    model_location: str,
    tokenizer: "PreTrainedTokenizerBase"
) -> str:
    """Process modelfile template with proper escaping."""
    # Special replacers to protect placeholders
    FILE_LOCATION_REPLACER = "⚫@✅#🦥__FILE_LOCATION__⚡@🦥#⛵"
    EOS_TOKEN_REPLACER = "⚫@✅#🦥__EOS_TOKEN__⚡@🦥#⛵"
    LEFT_BRACKET_REPLACER = "⚫@✅#🦥"
    RIGHT_BRACKET_REPLACER = "⚡@🦥#⛵"

    # Protect placeholders and escape brackets
    modelfile = (
        modelfile.replace("{__FILE_LOCATION__}", FILE_LOCATION_REPLACER)
        .replace("{__EOS_TOKEN__}", EOS_TOKEN_REPLACER)
        .replace("{", LEFT_BRACKET_REPLACER)
        .replace("}", RIGHT_BRACKET_REPLACER)
    )

    # Restore placeholders
    modelfile = modelfile.replace(
        FILE_LOCATION_REPLACER, "{__FILE_LOCATION__}"
    ).replace(EOS_TOKEN_REPLACER, "{__EOS_TOKEN__}")

    # Format with actual values
    if "__EOS_TOKEN__" in modelfile:
        modelfile = modelfile.format(
            __FILE_LOCATION__=model_location,
            __EOS_TOKEN__=tokenizer.eos_token,
        )
    else:
        modelfile = modelfile.format(
            __FILE_LOCATION__=model_location,
        )

    # Restore brackets
    modelfile = (
        modelfile.replace("⚫@✅#🦥", "{")
        .replace("⚡@🦥#⛵", "}")
        .rstrip()
    )

    return modelfile


def create_ollama_model(
    username: str,
    model_name: str,
    tag: str,
    modelfile_path: str
) -> Optional[str]:
    """
    Create an Ollama model from a Modelfile.

    Args:
        username: Ollama username
        model_name: Model name
        tag: Model tag
        modelfile_path: Path to Modelfile

    Returns:
        Error message or None on success
    """
    # Check if Ollama server is running
    try:
        init_check = subprocess.run(
            ["curl", "http://localhost:11434"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if init_check.returncode == 0:
            print(init_check.stdout.strip())
        else:
            print("Ollama Server is not Running")
            return "Server not running"
    except subprocess.TimeoutExpired:
        return "Ollama Request Timeout"

    # Create model
    process = subprocess.Popen(
        [
            "ollama",
            "create",
            f"{username}/{model_name}:{tag}",
            "-f",
            f"{modelfile_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()

    return_code = process.wait()

    if return_code != 0:
        print(f"\nMODEL CREATED FAILED WITH RETURN CODE {return_code}")
        return f"Failed with code {return_code}"
    else:
        print("\nMODEL CREATED SUCCESSFULLY")
        return None


def push_to_ollama_hub(
    username: str,
    model_name: str,
    tag: str
) -> Optional[str]:
    """
    Push model to Ollama Hub.

    Args:
        username: Ollama username
        model_name: Model name
        tag: Model tag

    Returns:
        Error message or None on success
    """
    # Check if Ollama server is running
    try:
        init_check = subprocess.run(
            ["curl", "http://localhost:11434"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if init_check.returncode == 0:
            print(init_check.stdout.strip())
        else:
            print("Ollama Server is not Running")
            return "Server not running"
    except subprocess.TimeoutExpired:
        return "Ollama Request Timeout"

    # Push model
    process = subprocess.Popen(
        ["ollama", "push", f"{username}/{model_name}:{tag}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        sys.stdout.flush()

    return_code = process.wait()

    if return_code != 0:
        print(f"\nMODEL PUBLISHED FAILED WITH RETURN CODE {return_code}")
        return f"Failed with code {return_code}"
    else:
        print("\nMODEL PUBLISHED SUCCESSFULLY")
        return None


def push_to_ollama(
    tokenizer: "PreTrainedTokenizerBase",
    gguf_location: str,
    username: str,
    model_name: str,
    tag: str
) -> bool:
    """
    Create and push model to Ollama.

    Args:
        tokenizer: Tokenizer for modelfile
        gguf_location: Path to GGUF file
        username: Ollama username
        model_name: Model name
        tag: Model tag

    Returns:
        True on success, False on failure
    """
    from unsloth.models.loader_utils import get_model_name

    # Get base model name from tokenizer
    if hasattr(tokenizer, "name_or_path"):
        base_model_name = get_model_name(tokenizer.name_or_path, load_in_4bit=False)
    else:
        base_model_name = model_name

    # Create modelfile
    model_file = create_ollama_modelfile(
        tokenizer=tokenizer,
        base_model_name=base_model_name,
        model_location=gguf_location
    )

    if model_file is None:
        print("Unsloth: Could not create Ollama modelfile")
        return False

    # Write modelfile
    modelfile_path = f"Modelfile_{model_name}"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(model_file)

    # Create model
    error = create_ollama_model(
        username=username,
        model_name=model_name,
        tag=tag,
        modelfile_path=modelfile_path,
    )

    if error:
        return False

    # Push to hub
    error = push_to_ollama_hub(
        username=username,
        model_name=model_name,
        tag=tag
    )

    if error:
        return False

    print("Successfully pushed to ollama")
    return True


def fix_tokenizer_bos_token(
    tokenizer: "PreTrainedTokenizerBase"
) -> tuple:
    """
    Fix BOS token in tokenizer chat template.

    Args:
        tokenizer: Tokenizer to fix

    Returns:
        Tuple of (fix_bos_token, old_chat_template)
    """
    from transformers.models.llama.modeling_llama import logger

    fix_bos_token = False
    chat_template = getattr(tokenizer, "chat_template", None)

    if tokenizer("A").input_ids[0] == getattr(tokenizer, "bos_token_id", None):
        if chat_template is not None and (
            tokenizer.bos_token in chat_template
            or "{bos_token}" in chat_template.replace(" ", "")
            or "{bos_token+" in chat_template.replace(" ", "")
        ):
            fix_bos_token = True
            logger.warning(
                "Unsloth: ##### The current model auto adds a BOS token.\n"
                "Unsloth: ##### Your chat template has a BOS token. We shall remove it temporarily."
            )

            # Remove {{bos_token}}
            new_chat_template = re.sub(
                r"\{[\s]{0,}\{[\s]{0,}bos\_token[\s]{0,}\}[\s]{0,}\}",
                "",
                chat_template
            )
            # Remove {{bos_token +
            new_chat_template = re.sub(
                r"\{[\s]{0,}\{[\s]{0,}bos\_token[\s]{0,}\+[\s]{0,}",
                "",
                new_chat_template,
            )

            tokenizer.chat_template = new_chat_template

    return fix_bos_token, chat_template
