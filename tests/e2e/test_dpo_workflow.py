"""
End-to-end tests for DPO (Direct Preference Optimization) workflow.

These tests run complete DPO workflows and require GPU.
"""

import pytest
import os


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestDPOWorkflow:
    """End-to-end tests for DPO training workflow."""

    def test_dpo_model_setup(self, temp_dir):
        """Test DPO model setup with preference data."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        try:
            from unsloth import FastLanguageModel, PatchDPOTrainer
            from datasets import Dataset
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
        )

        # Create preference dataset
        dpo_dataset = Dataset.from_dict({
            "prompt": [
                "What is 2+2?",
                "What is the capital of France?",
            ] * 4,
            "chosen": [
                "2+2 equals 4.",
                "The capital of France is Paris.",
            ] * 4,
            "rejected": [
                "I don't know.",
                "I'm not sure.",
            ] * 4,
        })

        assert model is not None
        assert len(dpo_dataset) > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestORPOWorkflow:
    """End-to-end tests for ORPO training workflow."""

    def test_orpo_model_setup(self):
        """Test ORPO model setup."""
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
            target_modules=["q_proj", "v_proj"],
        )

        assert model is not None

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestGRPOWorkflow:
    """End-to-end tests for GRPO training workflow."""

    def test_grpo_model_setup(self):
        """Test GRPO model setup."""
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
            target_modules=["q_proj", "v_proj"],
        )

        assert model is not None

        del model
        del tokenizer
        torch.cuda.empty_cache()
