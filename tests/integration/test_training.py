"""
Integration tests for model training.

These tests require GPU and will be skipped if CUDA is not available.
"""

import pytest
import os


@pytest.mark.gpu
@pytest.mark.integration
class TestTrainingSetup:
    """Tests for training setup."""

    def test_prepare_model_for_training(self):
        """Test preparing model for training."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        # Model should be in training mode
        model.train()
        assert model.training

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing setup."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            use_gradient_checkpointing="unsloth",
        )

        assert model is not None

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
class TestBasicTraining:
    """Tests for basic training operations."""

    def test_single_training_step(self):
        """Test a single training step."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

        # Prepare input
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        inputs["labels"] = inputs["input_ids"].clone()

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        assert loss is not None
        assert not loss.isnan()

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert not param.grad.isnan().any()
                break

        assert has_grad, "No gradients computed"

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_optimizer_step(self):
        """Test optimizer step."""
        import torch
        from torch.optim import AdamW

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
        )

        # Setup optimizer
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        # Training step
        text = "Test input"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Store initial params
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                initial_params[name] = param.clone()

        optimizer.step()
        optimizer.zero_grad()

        # Check params changed
        for name, param in model.named_parameters():
            if name in initial_params:
                assert not torch.equal(param, initial_params[name]), \
                    f"Parameter {name} did not change"
                break

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
class TestTrainerIntegration:
    """Tests for trainer integration."""

    def test_unsloth_trainer_setup(self, sample_training_dataset, temp_dir):
        """Test UnslothTrainer setup."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
        except ImportError:
            pytest.skip("UnslothTrainer not available")

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
        )

        training_args = UnslothTrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            max_steps=1,
            logging_steps=1,
            save_strategy="no",
        )

        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=sample_training_dataset,
        )

        assert trainer is not None

        del model
        del tokenizer
        del trainer
        torch.cuda.empty_cache()

    def test_short_training_run(self, sample_training_dataset, temp_dir):
        """Test a short training run."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
        except ImportError:
            pytest.skip("UnslothTrainer not available")

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
        )

        training_args = UnslothTrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            max_steps=2,
            logging_steps=1,
            save_strategy="no",
        )

        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=sample_training_dataset,
        )

        # Run training
        result = trainer.train()

        assert result is not None
        assert trainer.state.global_step >= 1

        del model
        del tokenizer
        del trainer
        torch.cuda.empty_cache()
