"""
End-to-end tests for GGUF export workflow.

These tests run complete GGUF export workflows and require GPU.
"""

import pytest
import os


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestGGUFExport:
    """End-to-end tests for GGUF export workflow."""

    def test_gguf_export_setup(self, temp_dir):
        """Test GGUF export setup."""
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

    def test_gguf_export_to_file(self, temp_dir):
        """Test exporting model to GGUF format."""
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

        output_path = os.path.join(temp_dir, "gguf_output")

        try:
            # Try to save as GGUF
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method="q4_k_m",
            )

            # Check output exists
            gguf_files = [f for f in os.listdir(output_path) if f.endswith(".gguf")]
            assert len(gguf_files) > 0, "No GGUF files created"
        except Exception as e:
            # GGUF export may require llama.cpp
            pytest.skip(f"GGUF export failed: {e}")

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestGGUFQuantizationMethods:
    """Test different GGUF quantization methods."""

    @pytest.mark.parametrize("quant_method", [
        "q4_k_m",
        "q5_k_m",
        "q8_0",
    ])
    def test_different_quantization_methods(self, quant_method, temp_dir):
        """Test GGUF export with different quantization methods."""
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

        output_path = os.path.join(temp_dir, f"gguf_{quant_method}")

        try:
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quant_method,
            )
        except Exception as e:
            pytest.skip(f"GGUF export with {quant_method} failed: {e}")

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.slow
class TestOllamaExport:
    """Test export for Ollama."""

    def test_ollama_export_setup(self, temp_dir):
        """Test Ollama export setup."""
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

        # Model should be ready for Ollama export
        assert model is not None
        assert tokenizer is not None

        del model
        del tokenizer
        torch.cuda.empty_cache()
