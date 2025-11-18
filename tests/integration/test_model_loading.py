"""
Integration tests for model loading.

These tests require GPU and will be skipped if CUDA is not available.
"""

import pytest


@pytest.mark.gpu
@pytest.mark.integration
class TestModelLoading:
    """Tests for model loading functionality."""

    def test_load_tinyllama_4bit(self):
        """Test loading TinyLlama 4-bit model."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        assert model is not None
        assert tokenizer is not None

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_model_config_loaded(self):
        """Test that model configuration is properly loaded."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        assert hasattr(model, "config")
        assert model.config is not None
        assert model.config.hidden_size > 0
        assert model.config.num_attention_heads > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_tokenizer_loaded_correctly(self):
        """Test that tokenizer is properly loaded."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token is not None or tokenizer.eos_token is not None

        # Test basic tokenization
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_model_on_gpu(self):
        """Test that model is loaded on GPU."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        # Check model is on GPU
        device = next(model.parameters()).device
        assert device.type == "cuda"

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_custom_max_seq_length(self):
        """Test loading with custom max sequence length."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        max_seq_length = 512
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )

        assert model is not None

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
class TestLoRAApplication:
    """Tests for LoRA application to models."""

    def test_apply_lora_to_model(self):
        """Test applying LoRA adapters to model."""
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
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        assert model is not None
        # Check that model has LoRA parameters
        lora_params = [n for n, p in model.named_parameters() if "lora" in n.lower()]
        assert len(lora_params) > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_lora_trainable_parameters(self):
        """Test that only LoRA parameters are trainable."""
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

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # Trainable should be much smaller than total
        assert trainable < total
        assert trainable > 0

        del model
        del tokenizer
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.integration
class TestModelInference:
    """Tests for model inference."""

    def test_model_forward_pass(self):
        """Test model forward pass."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        # Prepare input
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs is not None
        assert outputs.logits is not None
        assert not outputs.logits.isnan().any()

        del model
        del tokenizer
        torch.cuda.empty_cache()

    def test_model_generation(self):
        """Test model text generation."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        # Prepare input
        text = "The capital of France is"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

        assert outputs is not None
        assert outputs.shape[1] > inputs["input_ids"].shape[1]

        # Decode
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(decoded) > len(text)

        del model
        del tokenizer
        torch.cuda.empty_cache()
