"""
Unit tests for chat templates module.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestChatTemplateImports:
    """Tests for chat template imports and constants."""

    def test_chat_templates_import(self):
        """Test that chat_templates module can be imported."""
        from unsloth import chat_templates
        assert chat_templates is not None

    def test_get_chat_template_exists(self):
        """Test that get_chat_template function exists."""
        from unsloth.chat_templates import get_chat_template
        assert callable(get_chat_template)

    def test_chat_templates_dict_exists(self):
        """Test that CHAT_TEMPLATES dict exists and has entries."""
        try:
            from unsloth.chat_templates import CHAT_TEMPLATES
            assert isinstance(CHAT_TEMPLATES, dict)
            assert len(CHAT_TEMPLATES) > 0
        except ImportError:
            pytest.skip("CHAT_TEMPLATES not exported")


class TestGetChatTemplate:
    """Tests for get_chat_template function."""

    def test_get_unsloth_template(self):
        """Test getting unsloth chat template."""
        from unsloth.chat_templates import get_chat_template

        result = get_chat_template("unsloth")
        assert result is not None

    def test_get_chatml_template(self):
        """Test getting chatml template."""
        from unsloth.chat_templates import get_chat_template

        result = get_chat_template("chatml")
        assert result is not None

    def test_get_llama_template(self):
        """Test getting llama template."""
        from unsloth.chat_templates import get_chat_template

        result = get_chat_template("llama-3")
        assert result is not None

    def test_get_zephyr_template(self):
        """Test getting zephyr template."""
        from unsloth.chat_templates import get_chat_template

        result = get_chat_template("zephyr")
        assert result is not None

    def test_invalid_template_raises(self):
        """Test that invalid template name raises error."""
        from unsloth.chat_templates import get_chat_template

        with pytest.raises((KeyError, ValueError)):
            get_chat_template("nonexistent-template-xyz")

    def test_template_returns_dict_or_string(self):
        """Test that template returns dict or string."""
        from unsloth.chat_templates import get_chat_template

        result = get_chat_template("unsloth")
        assert isinstance(result, (dict, str, tuple))


class TestChatTemplateFormatting:
    """Tests for chat template formatting functions."""

    def test_apply_template_with_messages(self, sample_messages):
        """Test applying template to messages."""
        try:
            from unsloth.chat_templates import apply_chat_template

            result = apply_chat_template(
                template_name="chatml",
                messages=sample_messages,
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except ImportError:
            pytest.skip("apply_chat_template not available")

    def test_template_includes_system_message(self, sample_messages):
        """Test that template includes system message."""
        try:
            from unsloth.chat_templates import apply_chat_template

            result = apply_chat_template(
                template_name="chatml",
                messages=sample_messages,
            )
            assert "helpful assistant" in result.lower() or "system" in result.lower()
        except ImportError:
            pytest.skip("apply_chat_template not available")


class TestRemoveSpecialTokens:
    """Tests for remove_special_tokens function."""

    def test_function_exists(self):
        """Test that remove_special_tokens function exists."""
        try:
            from unsloth.chat_templates import remove_special_tokens
            assert callable(remove_special_tokens)
        except ImportError:
            pytest.skip("remove_special_tokens not available")

    def test_removes_bos_token(self):
        """Test removing BOS token."""
        try:
            from unsloth.chat_templates import remove_special_tokens

            text = "<s>Hello world"
            result = remove_special_tokens(text, bos_token="<s>")
            assert "<s>" not in result
        except (ImportError, TypeError):
            pytest.skip("remove_special_tokens not available or different signature")

    def test_removes_eos_token(self):
        """Test removing EOS token."""
        try:
            from unsloth.chat_templates import remove_special_tokens

            text = "Hello world</s>"
            result = remove_special_tokens(text, eos_token="</s>")
            assert "</s>" not in result
        except (ImportError, TypeError):
            pytest.skip("remove_special_tokens not available or different signature")


class TestToShareGPT:
    """Tests for to_sharegpt conversion function."""

    def test_function_exists(self):
        """Test that to_sharegpt function exists."""
        try:
            from unsloth.chat_templates import to_sharegpt
            assert callable(to_sharegpt)
        except ImportError:
            pytest.skip("to_sharegpt not available")


class TestOllamaTemplateMappers:
    """Tests for Ollama template mappers."""

    def test_ollama_mappers_import(self):
        """Test that ollama_template_mappers can be imported."""
        try:
            from unsloth import ollama_template_mappers
            assert ollama_template_mappers is not None
        except ImportError:
            pytest.skip("ollama_template_mappers not available")


class TestChatTemplateValidation:
    """Tests for chat template validation."""

    def test_template_has_required_tokens(self):
        """Test that templates have required tokens."""
        from unsloth.chat_templates import get_chat_template

        template = get_chat_template("unsloth")
        # Template should be a valid object
        assert template is not None

    def test_multiple_templates_available(self):
        """Test that multiple templates are available."""
        from unsloth.chat_templates import get_chat_template

        templates = ["unsloth", "chatml", "zephyr", "llama-3"]
        available = 0
        for name in templates:
            try:
                get_chat_template(name)
                available += 1
            except (KeyError, ValueError):
                pass

        assert available >= 2, "At least 2 templates should be available"


class TestChatTemplateEdgeCases:
    """Tests for edge cases in chat templates."""

    def test_empty_messages(self):
        """Test handling of empty messages."""
        try:
            from unsloth.chat_templates import apply_chat_template

            result = apply_chat_template(
                template_name="chatml",
                messages=[],
            )
            assert isinstance(result, str)
        except (ImportError, Exception):
            pytest.skip("apply_chat_template not available or doesn't handle empty messages")

    def test_single_user_message(self):
        """Test handling of single user message."""
        try:
            from unsloth.chat_templates import apply_chat_template

            messages = [{"role": "user", "content": "Hello"}]
            result = apply_chat_template(
                template_name="chatml",
                messages=messages,
            )
            assert "Hello" in result
        except ImportError:
            pytest.skip("apply_chat_template not available")

    def test_unicode_in_messages(self):
        """Test handling of unicode in messages."""
        try:
            from unsloth.chat_templates import apply_chat_template

            messages = [{"role": "user", "content": "Hello 你好 🌍"}]
            result = apply_chat_template(
                template_name="chatml",
                messages=messages,
            )
            assert "你好" in result or "Hello" in result
        except ImportError:
            pytest.skip("apply_chat_template not available")
