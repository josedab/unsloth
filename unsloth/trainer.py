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
Training utilities for Unsloth optimized models.

This module provides custom trainer classes and training utilities that work
seamlessly with Unsloth-optimized models. It includes support for different
learning rates for embeddings, gradient accumulation fixes, and backward
compatibility with different TRL versions.

The primary components are:

- :class:`UnslothTrainer`: Custom SFTTrainer with embedding learning rate support
- :class:`UnslothTrainingArguments`: Extended training arguments
- :func:`unsloth_train`: Wrapper function for training with gradient accumulation fixes

Example:
    Basic training setup::

        from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, ...)

        # Configure training
        training_args = UnslothTrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            embedding_learning_rate=1e-5,  # Lower LR for embeddings
        )

        # Create trainer and train
        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

See Also:
    - :mod:`unsloth.models.loader`: For loading models
    - :mod:`unsloth.save`: For saving trained models
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
]

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version

if Version(transformers_version) > Version("4.45.2"):

    def unsloth_train(trainer, *args, **kwargs):
        """
        Train a model using the Unsloth-optimized training loop.

        This function wraps the trainer's train method, automatically applying
        gradient accumulation fixes for older transformers versions.

        Args:
            trainer: An UnslothTrainer or SFTTrainer instance.
            *args: Additional positional arguments passed to trainer.train().
            **kwargs: Additional keyword arguments passed to trainer.train().

        Returns:
            The training output from trainer.train().

        Example:
            >>> trainer = UnslothTrainer(model=model, args=args, ...)
            >>> stats = unsloth_train(trainer)

        Note:
            For transformers > 4.45.2, this simply calls trainer.train().
            For older versions, it applies gradient accumulation fixes.
        """
        return trainer.train(*args, **kwargs)

else:

    def unsloth_train(trainer, *args, **kwargs):
        """
        Train a model using the Unsloth-optimized training loop.

        This function wraps the trainer's train method, automatically applying
        gradient accumulation fixes for older transformers versions.

        Args:
            trainer: An UnslothTrainer or SFTTrainer instance.
            *args: Not supported in older transformers versions.
            **kwargs: Not supported in older transformers versions.

        Returns:
            The training output from the custom training loop.

        Raises:
            RuntimeError: If args or kwargs are provided with older transformers.

        Note:
            For transformers <= 4.45.2, additional arguments are not supported.
            Consider upgrading transformers to use the full feature set.
        """
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
                "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"
            "`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`"
        )
        return _unsloth_train(trainer)


try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments


class UnslothTrainingArguments(TrainingArguments):
    """
    Extended training arguments with support for embedding-specific learning rates.

    This class extends the standard TrainingArguments (or SFTConfig for TRL >= 0.13)
    to add support for setting a different learning rate for embedding layers.
    This is useful when training models with newly added tokens.

    Args:
        embedding_learning_rate: Learning rate specifically for embedding layers.
            If None, uses the same learning rate as other parameters.
            Typically set lower than the main learning rate (e.g., 1e-5 vs 2e-4).
        *args: Additional positional arguments passed to TrainingArguments.
        **kwargs: Additional keyword arguments passed to TrainingArguments.

    Example:
        >>> from unsloth import UnslothTrainingArguments
        >>> args = UnslothTrainingArguments(
        ...     output_dir="./output",
        ...     learning_rate=2e-4,
        ...     embedding_learning_rate=1e-5,
        ...     per_device_train_batch_size=2,
        ...     gradient_accumulation_steps=4,
        ...     num_train_epochs=3,
        ... )

    Note:
        The embedding_learning_rate only takes effect when using UnslothTrainer
        and when embeddings are marked as trainable (e.g., after adding new tokens).

    See Also:
        - :class:`UnslothTrainer`: Trainer that uses these arguments
        - :func:`FastLanguageModel.get_peft_model`: For adding trainable embeddings
    """
    def __init__(self, embedding_learning_rate: float = None, *args, **kwargs):
        embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    """
    Create an optimizer with separate learning rates for embeddings.

    This internal function creates an optimizer that applies a different
    learning rate to embedding parameters compared to other model parameters.
    Embedding parameters are identified by the suffix "modules_to_save.default.weight".

    Args:
        model: The model to create an optimizer for.
        optimizer_cls: The optimizer class to instantiate (e.g., AdamW).
        optimizer_kwargs: Keyword arguments for the optimizer including 'lr'.
        embedding_lr: Learning rate for embedding parameters. Default: 5e-5.

    Returns:
        An optimizer instance with parameter groups for embeddings and non-embeddings.

    Note:
        This is an internal function used by UnslothTrainer.create_optimizer().
    """
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = {
        "non_embeddings": {},
        "embeddings": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[: -len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".") + 1 :]
            print(
                f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
            )
            param_groups["embeddings"][name] = param
        else:
            param_groups["non_embeddings"][name] = param

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["non_embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embeddings"].values()),
            "weight_decay": weight_decay,
            "lr": embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


class UnslothTrainer(SFTTrainer):
    """
    Custom trainer with support for embedding-specific learning rates.

    UnslothTrainer extends TRL's SFTTrainer to support different learning rates
    for embedding layers. This is useful when finetuning models with newly added
    tokens, where embeddings should be trained with a lower learning rate.

    The trainer automatically detects embedding parameters (those ending with
    "modules_to_save.default.weight") and applies the embedding_learning_rate
    specified in UnslothTrainingArguments.

    Example:
        >>> from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
        >>>
        >>> # Load and prepare model
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> model = FastLanguageModel.get_peft_model(model, ...)
        >>>
        >>> # Configure training with embedding LR
        >>> args = UnslothTrainingArguments(
        ...     output_dir="./output",
        ...     learning_rate=2e-4,
        ...     embedding_learning_rate=1e-5,
        ... )
        >>>
        >>> # Create trainer
        >>> trainer = UnslothTrainer(
        ...     model=model,
        ...     args=args,
        ...     train_dataset=dataset,
        ...     tokenizer=tokenizer,
        ... )
        >>> trainer.train()

    See Also:
        - :class:`UnslothTrainingArguments`: Training arguments with embedding_learning_rate
        - :func:`unsloth_train`: Wrapper for training with gradient accumulation fixes
    """

    def create_optimizer(self):
        """
        Create optimizer with optional separate embedding learning rate.

        This method overrides the default optimizer creation to support
        different learning rates for embedding parameters when
        embedding_learning_rate is specified in the training arguments.

        Returns:
            The optimizer instance, either standard or with embedding parameter groups.
        """
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer


# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _backwards_compatible_trainer(trainer_class, config_class):
    """
    Create a backwards-compatible trainer initializer for TRL version changes.

    TRL >= 0.13.0 changed how parameters are passed to trainers, moving many
    arguments from the trainer constructor to config classes. This function
    creates a wrapper that automatically handles the transition.

    Args:
        trainer_class: The TRL trainer class to patch (e.g., SFTTrainer).
        config_class: The corresponding config class (e.g., SFTConfig).

    Returns:
        A new __init__ method that handles both old and new TRL API styles.

    Note:
        This is an internal function used by _patch_trl_trainer().
    """
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove("self")
            trainer_params.remove("args")

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field
                for field in dataclasses.fields(config_class)
                if field.init
            }

            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments

            moved_params = set(inspect.signature(config_class).parameters.keys()) - set(
                inspect.signature(TrainingArguments).parameters.keys()
            )

            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params:
                    trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        original_init(self, *args, **kwargs)

    return new_init


def _patch_trl_trainer():
    """
    Patch TRL trainers for backwards compatibility across versions.

    This function patches all TRL trainer classes to support both old and new
    API styles, allowing users to use the same code across different TRL versions.
    It's called automatically when importing unsloth.

    The function:
    1. Finds all TRL trainer/config pairs
    2. Patches each trainer's __init__ to handle both old and new argument styles
    3. Sets a flag to prevent double-patching

    Note:
        - Only patches TRL versions > 0.11.0
        - Sets trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True when complete
        - Safe to call multiple times (no-op if already patched)

    Example:
        >>> from unsloth.trainer import _patch_trl_trainer
        >>> _patch_trl_trainer()  # Now TRL trainers accept both old and new APIs
    """
    import trl

    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl.__version__) <= Version("0.11.0"):
        return

    import trl.trainer

    trl_classes = dir(trl.trainer)
    trl_trainers = set(
        x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer")
    )
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:
            exec(
                f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
                globals(),
            )
        except:
            continue

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
