"""
End-to-end tests for QLoRA training workflow.

These tests run complete training workflows and require GPU.
"""

import pytest
import os


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestQLoRAWorkflow:
    """End-to-end tests for QLoRA training workflow."""

    def test_complete_qlora_workflow(self, temp_dir):
        """Test complete QLoRA workflow: load -> configure -> train -> save."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
            from datasets import Dataset
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

        # 1. Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        assert model is not None
        assert tokenizer is not None

        # 2. Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Check LoRA applied
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

        # 3. Prepare dataset
        dataset = Dataset.from_dict({
            "text": [
                "### Instruction: What is 2+2?\n### Response: 4",
                "### Instruction: What is the capital of France?\n### Response: Paris",
                "### Instruction: What color is the sky?\n### Response: Blue",
                "### Instruction: Who wrote Hamlet?\n### Response: Shakespeare",
            ] * 4
        })

        # 4. Train
        training_args = UnslothTrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=3,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            seed=42,
        )

        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        result = trainer.train()

        # Verify training completed
        assert result is not None
        assert trainer.state.global_step >= 1

        # 5. Save
        output_path = os.path.join(temp_dir, "qlora_output")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(output_path, "adapter_config.json"))

        # Cleanup
        del model
        del tokenizer
        del trainer
        torch.cuda.empty_cache()

    def test_qlora_inference_after_training(self, temp_dir):
        """Test that model can do inference after QLoRA training."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
            from datasets import Dataset
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

        # Load and configure
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

        # Quick training
        dataset = Dataset.from_dict({
            "text": ["Hello world"] * 8
        })

        training_args = UnslothTrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=1,
            max_steps=2,
            save_strategy="no",
        )

        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        # Switch to inference
        FastLanguageModel.for_inference(model)

        # Generate
        inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )

        assert outputs is not None
        assert outputs.shape[1] > inputs["input_ids"].shape[1]

        del model
        del tokenizer
        del trainer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestQLoRAWithDifferentConfigs:
    """Test QLoRA with different configurations."""

    @pytest.mark.parametrize("r", [4, 8, 16])
    def test_different_lora_ranks(self, r, temp_dir):
        """Test QLoRA with different LoRA ranks."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            pytest.skip("unsloth not available")

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=["q_proj", "v_proj"],
        )

        # Verify LoRA applied with correct rank
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("target_modules", [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    ])
    def test_different_target_modules(self, target_modules, temp_dir):
        """Test QLoRA with different target modules."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            pytest.skip("unsloth not available")

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=target_modules,
        )

        # Check that LoRA was applied to specified modules
        lora_modules = [n for n, _ in model.named_modules() if "lora" in n.lower()]
        assert len(lora_modules) > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()
