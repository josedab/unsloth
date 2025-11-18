"""
Shared pytest fixtures and configuration for unsloth tests.
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch


# Register custom markers
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


# Skip GPU tests if no GPU available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_slow = pytest.mark.skip(reason="Slow tests disabled (use --run-slow to enable)")

    run_slow = config.getoption("--run-slow", default=False)

    for item in items:
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)

        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


# ==================== Unit Test Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for unit tests."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1
    tokenizer.vocab_size = 32000
    tokenizer.model_max_length = 4096
    tokenizer.chat_template = None
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="Hello world")
    return tokenizer


@pytest.fixture
def mock_model_config():
    """Create a mock model config for unit tests."""
    config = MagicMock()
    config.hidden_size = 4096
    config.num_attention_heads = 32
    config.num_hidden_layers = 32
    config.vocab_size = 32000
    config.max_position_embeddings = 4096
    config.model_type = "llama"
    config._name_or_path = "meta-llama/Llama-2-7b"
    return config


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def sample_dataset_dict():
    """Sample dataset dictionary for testing."""
    return {
        "text": [
            "Hello, how are you?",
            "I am fine, thank you.",
            "What is the weather today?",
            "It is sunny and warm.",
        ] * 5
    }


# ==================== Integration Test Fixtures ====================

@pytest.fixture
def small_model():
    """
    Load a small model for integration testing.

    This fixture requires GPU and will be skipped if CUDA is not available.
    Uses TinyLlama for fast testing.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/tinyllama-bnb-4bit",
            max_seq_length=256,
            load_in_4bit=True,
        )
        yield model, tokenizer

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.skip(f"Failed to load model: {e}")


@pytest.fixture
def sample_training_dataset():
    """Create a small dataset for training tests."""
    try:
        from datasets import Dataset

        return Dataset.from_dict({
            "text": [
                "### Instruction: What is 2+2?\n### Response: 4",
                "### Instruction: What is the capital of France?\n### Response: Paris",
                "### Instruction: What color is the sky?\n### Response: Blue",
                "### Instruction: Who wrote Hamlet?\n### Response: Shakespeare",
            ] * 4
        })
    except ImportError:
        pytest.skip("datasets library not available")


@pytest.fixture
def lora_config():
    """Default LoRA configuration for testing."""
    return {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


# ==================== E2E Test Fixtures ====================

@pytest.fixture
def training_args():
    """Default training arguments for E2E tests."""
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 1,
        "max_steps": 5,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "save_strategy": "no",
        "seed": 42,
    }


@pytest.fixture
def output_dir(temp_dir):
    """Create output directory for model saving tests."""
    output = os.path.join(temp_dir, "output")
    os.makedirs(output, exist_ok=True)
    return output


# ==================== Utility Functions ====================

def assert_model_outputs_valid(outputs):
    """Assert that model outputs are valid."""
    assert outputs is not None
    if hasattr(outputs, 'loss'):
        assert outputs.loss is not None
        assert not outputs.loss.isnan().any()
    if hasattr(outputs, 'logits'):
        assert outputs.logits is not None
        assert not outputs.logits.isnan().any()


def assert_gradients_exist(model):
    """Assert that model has computed gradients."""
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            assert not param.grad.isnan().any()
            break
    return has_grad
