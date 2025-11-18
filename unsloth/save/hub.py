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

"""HuggingFace Hub operations for model saving."""

from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path
import os
import glob

from .base import SaveConfig, HubError
from .validation import validate_hub_token

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from huggingface_hub import HfApi

__all__ = [
    "push_to_hub",
    "create_huggingface_repo",
    "upload_to_huggingface",
    "determine_username",
    "MODEL_CARD",
]


MODEL_CARD = """---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""


def push_to_hub(
    model: "PreTrainedModel",
    tokenizer: Optional["PreTrainedTokenizerBase"],
    config: SaveConfig,
) -> str:
    """
    Push model to HuggingFace Hub.

    Args:
        model: Model to push
        tokenizer: Associated tokenizer
        config: Hub configuration

    Returns:
        Hub URL

    Raises:
        HubError: Push failed
    """
    from huggingface_hub import HfApi

    token = validate_hub_token(config.token, required=True)

    if not config.repo_id:
        raise HubError("repo_id required for Hub push")

    api = HfApi(token=token)

    # Create repo if needed
    repo_id = create_repo_if_needed(api, config, model)

    # Update model tag
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

    return f"https://huggingface.co/{repo_id}"


def create_repo_if_needed(
    api: "HfApi",
    config: SaveConfig,
    model: "PreTrainedModel"
) -> str:
    """
    Create HuggingFace repo if it doesn't exist.

    Args:
        api: HuggingFace API client
        config: Save configuration
        model: Model for metadata

    Returns:
        Full repo ID
    """
    from huggingface_hub import create_repo, ModelCard

    repo_id = config.repo_id or str(config.save_directory)
    token = config.token

    # Get username if not in repo_id
    if "/" not in repo_id:
        username = api.whoami()["name"]
        repo_id = f"{username}/{repo_id}"
    else:
        username = repo_id.split("/")[0]

    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            exist_ok=False,
            private=config.private,
        )

        # Create model card
        content = MODEL_CARD.format(
            username=username,
            base_model=model.config._name_or_path,
            model_type=model.config.model_type,
            method="",
            extra="unsloth",
        )
        card = ModelCard(content)
        card.push_to_hub(repo_id, token=token)
    except Exception:
        # Repo might already exist
        pass

    return repo_id


def create_huggingface_repo(
    model: "PreTrainedModel",
    save_directory: str,
    token: Optional[str] = None,
    private: bool = False,
) -> Tuple[str, "HfApi"]:
    """
    Create a HuggingFace repository.

    Args:
        model: Model for metadata
        save_directory: Repository name/path
        token: HuggingFace token
        private: Whether repo should be private

    Returns:
        Tuple of (repo_id, HfApi instance)
    """
    from huggingface_hub import HfApi, create_repo, ModelCard

    token = validate_hub_token(token)
    save_directory, username = determine_username(save_directory, "", token)

    try:
        create_repo(
            repo_id=save_directory,
            token=token,
            repo_type="model",
            exist_ok=False,
            private=private,
        )

        # Create model card
        content = MODEL_CARD.format(
            username=username,
            base_model=model.config._name_or_path,
            model_type=model.config.model_type,
            method="",
            extra="unsloth",
        )
        card = ModelCard(content)
        card.push_to_hub(save_directory, token=token)
    except Exception:
        pass

    api = HfApi(token=token)
    return save_directory, api


def upload_to_huggingface(
    model: "PreTrainedModel",
    save_directory: str,
    token: str,
    method: str,
    extra: str = "",
    file_location: Optional[str] = None,
    old_username: Optional[str] = None,
    private: Optional[bool] = None,
    create_config: bool = True,
) -> str:
    """
    Upload model files to HuggingFace Hub.

    Args:
        model: Model for metadata
        save_directory: Repository ID
        token: HuggingFace token
        method: Upload method description
        extra: Extra tag
        file_location: Specific file to upload
        old_username: Previous username
        private: Whether repo is private
        create_config: Whether to create config.json

    Returns:
        Username
    """
    from huggingface_hub import HfApi, create_repo, ModelCard
    import json

    save_directory, username = determine_username(save_directory, old_username, token)

    try:
        create_repo(
            repo_id=save_directory,
            token=token,
            repo_type="model",
            exist_ok=False,
            private=private,
        )

        # Create model card
        content = MODEL_CARD.format(
            username=username,
            base_model=model.config._name_or_path,
            model_type=model.config.model_type,
            method="",
            extra=extra,
        )
        card = ModelCard(content)
        card.push_to_hub(save_directory, token=token)
    except Exception:
        pass

    if file_location is not None:
        api = HfApi(token=token)

        if "/" in file_location:
            uploaded_location = file_location[file_location.rfind("/") + 1:]
        else:
            uploaded_location = file_location

        # Upload tensorboard files
        ftevent_files = glob.glob("*out.tfevents*", recursive=True)
        if ftevent_files:
            print(
                "Unsloth: Uploading tensorboard files...",
                file_location + "*out.tfevents*",
            )
            for ftevent_file in ftevent_files:
                api.upload_file(
                    path_or_fileobj=ftevent_file,
                    path_in_repo=ftevent_file.replace(file_location, ""),
                    repo_id=save_directory,
                    repo_type="model",
                    commit_message="(Trained with Unsloth)",
                )

        # Upload main file
        api.upload_file(
            path_or_fileobj=file_location,
            path_in_repo=uploaded_location,
            repo_id=save_directory,
            repo_type="model",
            commit_message="(Trained with Unsloth)",
        )

        # Upload config.json
        if create_config:
            config_file = "_temporary_unsloth_config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"model_type": model.config.model_type}, f, indent=4)

            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo="config.json",
                repo_id=save_directory,
                repo_type="model",
                commit_message="(Trained with Unsloth)",
            )
            os.remove(config_file)

    return username


def determine_username(
    save_directory: str,
    old_username: Optional[str],
    token: Optional[str]
) -> Tuple[str, str]:
    """
    Determine username from save directory.

    Args:
        save_directory: Save directory or repo ID
        old_username: Previous username
        token: HuggingFace token

    Returns:
        Tuple of (full_repo_id, username)

    Raises:
        HubError: Cannot determine username
    """
    from huggingface_hub import whoami

    username = ""
    save_directory = save_directory.lstrip("./")

    if "/" not in save_directory:
        try:
            username = whoami(token=token)["name"]
            if isinstance(old_username, str) and username != old_username:
                username = old_username
            save_directory = f"{username}/{save_directory}"
        except Exception as e:
            raise HubError(
                f"{save_directory} is not a valid HuggingFace directory: {e}"
            )
    else:
        username = save_directory.split("/")[0]

    return save_directory, username
