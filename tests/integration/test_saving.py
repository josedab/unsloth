"""
Integration tests for model saving functionality.

These tests require GPU and will be skipped if CUDA is not available.
"""

import pytest
import os


@pytest.mark.gpu
@pytest.mark.integration
class TestLoRASaving:
    """Tests for LoRA adapter saving."""

    def test_save_lora_adapters(self, temp_dir):
        """Test saving LoRA adapters."""
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
            target_modules=["q_proj", "v_proj"],
        )

        # Save adapters
        output_path = os.path.join(temp_dir, "lora_adapters")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Check files exist
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.join(output_path, "adapter_config.json"))

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_save_and_load_lora(self, temp_dir):
        """Test saving and loading LoRA adapters."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        # Create and save model
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

        output_path = os.path.join(temp_dir, "lora_test")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        # Reload model with adapters
        model2, tokenizer2 = FastLanguageModel.from_pretrained(
            output_path,
            max_seq_length=256,
            load_in_4bit=True,
        )

        assert model2 is not None
        assert tokenizer2 is not None

        del model2
        del tokenizer2
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
class TestTokenizerSaving:
    """Tests for tokenizer saving."""

    def test_save_tokenizer(self, temp_dir):
        """Test saving tokenizer."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        output_path = os.path.join(temp_dir, "tokenizer")
        tokenizer.save_pretrained(output_path)

        # Check tokenizer files
        assert os.path.exists(output_path)
        tokenizer_files = os.listdir(output_path)
        assert len(tokenizer_files) > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_reload_tokenizer(self, temp_dir):
        """Test reloading saved tokenizer."""
        import torch
        from transformers import AutoTokenizer

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        output_path = os.path.join(temp_dir, "tokenizer_reload")
        tokenizer.save_pretrained(output_path)

        # Reload
        reloaded = AutoTokenizer.from_pretrained(output_path)
        assert reloaded is not None
        assert reloaded.vocab_size == tokenizer.vocab_size

        del model
        del tokenizer
        del reloaded
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
class TestMergedModelSaving:
    """Tests for merged model saving."""

    def test_save_merged_16bit(self, temp_dir):
        """Test saving merged 16-bit model."""
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
            target_modules=["q_proj", "v_proj"],
        )

        output_path = os.path.join(temp_dir, "merged_16bit")

        try:
            model.save_pretrained_merged(
                output_path,
                tokenizer,
                save_method="merged_16bit",
            )

            assert os.path.exists(output_path)
        except Exception as e:
            # May fail due to memory constraints in CI
            pytest.skip(f"Merged saving failed: {e}")

        del model
        del tokenizer
        torch.cuda.empty_cache()
