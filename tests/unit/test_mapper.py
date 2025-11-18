"""
Unit tests for the model mapper module.
"""

import pytest

from unsloth.models.mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER


class TestIntToFloatMapper:
    """Tests for INT_TO_FLOAT_MAPPER."""

    def test_mapper_not_empty(self):
        """Test that mapper contains entries."""
        assert len(INT_TO_FLOAT_MAPPER) > 0

    def test_all_keys_are_strings(self):
        """Test that all keys are strings."""
        for key in INT_TO_FLOAT_MAPPER.keys():
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_all_values_are_tuples(self):
        """Test that all values are tuples."""
        for key, value in INT_TO_FLOAT_MAPPER.items():
            assert isinstance(value, tuple), f"Value for {key} is not a tuple"

    def test_all_tuples_contain_strings(self):
        """Test that all tuples contain only strings."""
        for key, values in INT_TO_FLOAT_MAPPER.items():
            for v in values:
                assert isinstance(v, str), f"Value {v} in {key} is not a string"

    def test_common_models_present(self):
        """Test that common models are in the mapper."""
        common_models = [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/tinyllama-bnb-4bit",
        ]
        for model in common_models:
            assert model in INT_TO_FLOAT_MAPPER, f"{model} not found in mapper"

    def test_tinyllama_mapping(self):
        """Test specific TinyLlama mapping."""
        key = "unsloth/tinyllama-bnb-4bit"
        assert key in INT_TO_FLOAT_MAPPER
        mappings = INT_TO_FLOAT_MAPPER[key]
        assert "unsloth/tinyllama" in mappings or "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" in mappings

    def test_llama3_mapping(self):
        """Test Llama 3 mapping."""
        key = "unsloth/llama-3-8b-bnb-4bit"
        assert key in INT_TO_FLOAT_MAPPER
        mappings = INT_TO_FLOAT_MAPPER[key]
        assert len(mappings) > 0

    def test_mistral_mapping(self):
        """Test Mistral mapping."""
        key = "unsloth/mistral-7b-bnb-4bit"
        assert key in INT_TO_FLOAT_MAPPER
        mappings = INT_TO_FLOAT_MAPPER[key]
        assert any("mistral" in m.lower() for m in mappings)

    def test_gemma_models_present(self):
        """Test that Gemma models are in mapper."""
        gemma_keys = [k for k in INT_TO_FLOAT_MAPPER.keys() if "gemma" in k.lower()]
        assert len(gemma_keys) > 0, "No Gemma models found in mapper"

    def test_qwen_models_present(self):
        """Test that Qwen models are in mapper."""
        qwen_keys = [k for k in INT_TO_FLOAT_MAPPER.keys() if "qwen" in k.lower()]
        assert len(qwen_keys) > 0, "No Qwen models found in mapper"

    def test_phi_models_present(self):
        """Test that Phi models are in mapper."""
        phi_keys = [k for k in INT_TO_FLOAT_MAPPER.keys() if "phi" in k.lower()]
        assert len(phi_keys) > 0, "No Phi models found in mapper"

    def test_keys_follow_naming_convention(self):
        """Test that keys follow expected naming convention."""
        for key in INT_TO_FLOAT_MAPPER.keys():
            # Should contain org/model format
            assert "/" in key, f"Key {key} doesn't contain '/'"
            parts = key.split("/")
            assert len(parts) == 2, f"Key {key} has unexpected format"

    def test_no_duplicate_values_in_tuples(self):
        """Test that tuples don't contain duplicates."""
        for key, values in INT_TO_FLOAT_MAPPER.items():
            assert len(values) == len(set(values)), f"Duplicate values in {key}"


class TestFloatToIntMapper:
    """Tests for FLOAT_TO_INT_MAPPER."""

    def test_mapper_not_empty(self):
        """Test that mapper contains entries."""
        assert len(FLOAT_TO_INT_MAPPER) > 0

    def test_all_keys_are_strings(self):
        """Test that all keys are strings."""
        for key in FLOAT_TO_INT_MAPPER.keys():
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_all_values_are_strings(self):
        """Test that all values are strings."""
        for key, value in FLOAT_TO_INT_MAPPER.items():
            assert isinstance(value, str), f"Value for {key} is not a string"

    def test_reverse_mapping_consistency(self):
        """Test that FLOAT_TO_INT_MAPPER is consistent with INT_TO_FLOAT_MAPPER."""
        for int_key, float_values in INT_TO_FLOAT_MAPPER.items():
            for float_val in float_values:
                if float_val in FLOAT_TO_INT_MAPPER:
                    # The float model should map back to the int model
                    assert FLOAT_TO_INT_MAPPER[float_val] == int_key, \
                        f"Inconsistent mapping: {float_val} -> {FLOAT_TO_INT_MAPPER[float_val]}, expected {int_key}"


class TestMapperCoverage:
    """Tests for mapper coverage of model families."""

    def test_llama_family_coverage(self):
        """Test coverage of Llama model family."""
        llama_models = [k for k in INT_TO_FLOAT_MAPPER.keys() if "llama" in k.lower()]
        assert len(llama_models) >= 5, "Expected at least 5 Llama models"

    def test_instruct_models_included(self):
        """Test that instruct models are included."""
        instruct_models = [k for k in INT_TO_FLOAT_MAPPER.keys() if "instruct" in k.lower()]
        assert len(instruct_models) > 0, "No instruct models found"

    def test_chat_models_included(self):
        """Test that chat models are included."""
        chat_models = [k for k in INT_TO_FLOAT_MAPPER.keys() if "chat" in k.lower()]
        assert len(chat_models) > 0, "No chat models found"

    def test_code_models_included(self):
        """Test that code models are included."""
        code_models = [k for k in INT_TO_FLOAT_MAPPER.keys() if "code" in k.lower()]
        assert len(code_models) > 0, "No code models found"


class TestMapperModelResolution:
    """Tests for resolving models through mapper."""

    def test_get_float_model_for_int(self):
        """Test getting float model for quantized model."""
        int_model = "unsloth/tinyllama-bnb-4bit"
        float_models = INT_TO_FLOAT_MAPPER.get(int_model, ())
        assert len(float_models) > 0

    def test_get_int_model_for_float(self):
        """Test getting quantized model for float model."""
        # Find a float model that should map back
        for int_key, float_values in INT_TO_FLOAT_MAPPER.items():
            for float_val in float_values:
                if float_val in FLOAT_TO_INT_MAPPER:
                    int_model = FLOAT_TO_INT_MAPPER[float_val]
                    assert int_model == int_key
                    return
        pytest.skip("No reverse mappings found")

    def test_unknown_model_returns_empty(self):
        """Test that unknown model returns empty tuple."""
        result = INT_TO_FLOAT_MAPPER.get("nonexistent/model", ())
        assert result == ()

    def test_model_lookup_performance(self):
        """Test that model lookup is fast (dict lookup)."""
        import time

        model = "unsloth/llama-3-8b-bnb-4bit"

        start = time.time()
        for _ in range(10000):
            _ = INT_TO_FLOAT_MAPPER.get(model)
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 10k lookups)
        assert elapsed < 0.1, f"Lookup too slow: {elapsed}s"
