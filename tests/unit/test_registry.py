"""
Unit tests for the model registry module.
"""

import pytest
from unittest.mock import patch, MagicMock

from unsloth.registry.registry import (
    QuantType,
    ModelInfo,
    ModelMeta,
    MODEL_REGISTRY,
    register_model,
    QUANT_TAG_MAP,
    BNB_QUANTIZED_TAG,
    UNSLOTH_DYNAMIC_QUANT_TAG,
)


class TestQuantType:
    """Tests for QuantType enum."""

    def test_quant_type_values(self):
        """Test that QuantType enum has expected values."""
        assert QuantType.BNB.value == "bnb"
        assert QuantType.UNSLOTH.value == "unsloth"
        assert QuantType.GGUF.value == "GGUF"
        assert QuantType.NONE.value == "none"
        assert QuantType.BF16.value == "bf16"

    def test_quant_tag_map(self):
        """Test that QUANT_TAG_MAP maps correctly."""
        assert QUANT_TAG_MAP[QuantType.BNB] == BNB_QUANTIZED_TAG
        assert QUANT_TAG_MAP[QuantType.UNSLOTH] == UNSLOTH_DYNAMIC_QUANT_TAG
        assert QUANT_TAG_MAP[QuantType.GGUF] == "GGUF"
        assert QUANT_TAG_MAP[QuantType.NONE] is None
        assert QUANT_TAG_MAP[QuantType.BF16] == "bf16"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        info = ModelInfo(
            org="unsloth",
            base_name="llama",
            version="3",
            size=8,
            quant_type=QuantType.BNB,
        )
        assert info.org == "unsloth"
        assert info.base_name == "llama"
        assert info.version == "3"
        assert info.size == 8
        assert info.quant_type == QuantType.BNB

    def test_model_path_property(self):
        """Test the model_path property."""
        info = ModelInfo(
            org="unsloth",
            base_name="llama",
            version="3",
            size=8,
            name="llama-3-8b-bnb-4bit",
            quant_type=QuantType.BNB,
        )
        assert info.model_path == "unsloth/llama-3-8b-bnb-4bit"

    def test_append_instruct_tag(self):
        """Test appending instruct tag to key."""
        result = ModelInfo.append_instruct_tag("model", "Instruct")
        assert result == "model-Instruct"

        # No tag
        result = ModelInfo.append_instruct_tag("model", None)
        assert result == "model"

    def test_append_quant_type(self):
        """Test appending quantization type to key."""
        result = ModelInfo.append_quant_type("model", QuantType.BNB)
        assert "bnb-4bit" in result

        # No quantization
        result = ModelInfo.append_quant_type("model", QuantType.NONE)
        assert result == "model"

    def test_model_info_with_custom_name(self):
        """Test ModelInfo with custom name."""
        info = ModelInfo(
            org="unsloth",
            base_name="llama",
            version="3",
            size=8,
            name="custom-model-name",
            quant_type=QuantType.NONE,
        )
        assert info.name == "custom-model-name"
        assert info.model_path == "unsloth/custom-model-name"

    def test_model_info_multimodal(self):
        """Test ModelInfo with multimodal flag."""
        info = ModelInfo(
            org="unsloth",
            base_name="llava",
            version="1.5",
            size=7,
            is_multimodal=True,
            quant_type=QuantType.BNB,
        )
        assert info.is_multimodal is True


class TestModelMeta:
    """Tests for ModelMeta dataclass."""

    def test_model_meta_creation(self):
        """Test creating a ModelMeta instance."""
        meta = ModelMeta(
            org="unsloth",
            base_name="llama",
            model_version="3",
            model_info_cls=ModelInfo,
            model_sizes=[8, 70],
            instruct_tags=["Instruct"],
            quant_types=[QuantType.BNB, QuantType.UNSLOTH],
        )
        assert meta.org == "unsloth"
        assert meta.base_name == "llama"
        assert 8 in meta.model_sizes
        assert 70 in meta.model_sizes
        assert "Instruct" in meta.instruct_tags


class TestRegisterModel:
    """Tests for register_model function."""

    def setup_method(self):
        """Clear registry before each test."""
        MODEL_REGISTRY.clear()

    def test_register_model_basic(self):
        """Test basic model registration."""
        register_model(
            model_info_cls=ModelInfo,
            org="test-org",
            base_name="test-model",
            version="1",
            size=7,
            quant_type=QuantType.BNB,
            name="test-model-7b-bnb-4bit",
        )

        assert "test-org/test-model-7b-bnb-4bit" in MODEL_REGISTRY
        info = MODEL_REGISTRY["test-org/test-model-7b-bnb-4bit"]
        assert info.org == "test-org"
        assert info.size == 7

    def test_register_duplicate_model_raises(self):
        """Test that registering duplicate model raises ValueError."""
        register_model(
            model_info_cls=ModelInfo,
            org="test-org",
            base_name="test-model",
            version="1",
            size=7,
            quant_type=QuantType.BNB,
            name="duplicate-model",
        )

        with pytest.raises(ValueError, match="already registered"):
            register_model(
                model_info_cls=ModelInfo,
                org="test-org",
                base_name="test-model",
                version="1",
                size=7,
                quant_type=QuantType.BNB,
                name="duplicate-model",
            )

    def test_register_model_with_instruct_tag(self):
        """Test registering model with instruct tag."""
        register_model(
            model_info_cls=ModelInfo,
            org="test-org",
            base_name="test-model",
            version="1",
            size=7,
            instruct_tag="Instruct",
            quant_type=QuantType.BNB,
            name="test-model-7b-Instruct-bnb-4bit",
        )

        key = "test-org/test-model-7b-Instruct-bnb-4bit"
        assert key in MODEL_REGISTRY
        assert MODEL_REGISTRY[key].instruct_tag == "Instruct"

    def test_register_multimodal_model(self):
        """Test registering multimodal model."""
        register_model(
            model_info_cls=ModelInfo,
            org="test-org",
            base_name="vision-model",
            version="1",
            size=7,
            is_multimodal=True,
            quant_type=QuantType.BNB,
            name="vision-model-7b-bnb-4bit",
        )

        key = "test-org/vision-model-7b-bnb-4bit"
        assert MODEL_REGISTRY[key].is_multimodal is True


class TestSearchModels:
    """Tests for search_models function."""

    def setup_method(self):
        """Set up test models in registry."""
        MODEL_REGISTRY.clear()

        # Register test models
        register_model(
            model_info_cls=ModelInfo,
            org="unsloth",
            base_name="llama",
            version="3",
            size=8,
            quant_type=QuantType.BNB,
            name="llama-3-8b-bnb-4bit",
        )
        register_model(
            model_info_cls=ModelInfo,
            org="unsloth",
            base_name="llama",
            version="3",
            size=70,
            quant_type=QuantType.BNB,
            name="llama-3-70b-bnb-4bit",
        )
        register_model(
            model_info_cls=ModelInfo,
            org="unsloth",
            base_name="qwen",
            version="2",
            size=7,
            quant_type=QuantType.UNSLOTH,
            name="qwen-2-7b-unsloth-bnb-4bit",
        )

    def test_search_all_models(self):
        """Test searching for all models."""
        from unsloth.registry import search_models

        results = search_models()
        assert len(results) == 3

    def test_search_by_org(self):
        """Test searching by organization."""
        from unsloth.registry import search_models

        results = search_models(org="unsloth")
        assert len(results) == 3

    def test_search_by_base_name(self):
        """Test searching by base model name."""
        from unsloth.registry import search_models

        results = search_models(base_name="llama")
        assert len(results) == 2
        assert all(m.base_name == "llama" for m in results)

    def test_search_by_quant_type(self):
        """Test searching by quantization type."""
        from unsloth.registry import search_models

        results = search_models(quant_types=[QuantType.BNB])
        assert len(results) == 2
        assert all(m.quant_type == QuantType.BNB for m in results)

    def test_search_by_size(self):
        """Test searching by model size."""
        from unsloth.registry import search_models

        results = search_models(size=8)
        assert len(results) == 1
        assert results[0].size == 8

    def test_search_by_pattern(self):
        """Test searching by pattern."""
        from unsloth.registry import search_models

        results = search_models(pattern="llama")
        assert len(results) == 2


class TestRegistryIntegration:
    """Integration tests for registry module."""

    def test_register_models_imports(self):
        """Test that register_models can be imported and called."""
        from unsloth.registry import register_models

        MODEL_REGISTRY.clear()
        register_models()

        # Should have registered many models
        assert len(MODEL_REGISTRY) > 0

    def test_all_registered_models_have_required_fields(self):
        """Test that all registered models have required fields."""
        from unsloth.registry import register_models

        MODEL_REGISTRY.clear()
        register_models()

        for key, model in MODEL_REGISTRY.items():
            assert model.org is not None, f"{key} missing org"
            assert model.base_name is not None, f"{key} missing base_name"
            assert model.version is not None, f"{key} missing version"
            assert model.size is not None, f"{key} missing size"
            assert model.quant_type is not None, f"{key} missing quant_type"
