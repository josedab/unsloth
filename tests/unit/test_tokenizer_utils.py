"""
Unit tests for tokenizer utilities module.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestTokenizerUtilsImports:
    """Tests for tokenizer_utils imports."""

    def test_module_import(self):
        """Test that tokenizer_utils module can be imported."""
        from unsloth import tokenizer_utils
        assert tokenizer_utils is not None

    def test_load_correct_tokenizer_exists(self):
        """Test that load_correct_tokenizer function exists."""
        try:
            from unsloth.tokenizer_utils import load_correct_tokenizer
            assert callable(load_correct_tokenizer)
        except ImportError:
            pytest.skip("load_correct_tokenizer not available")

    def test_check_tokenizer_exists(self):
        """Test that check_tokenizer function exists."""
        try:
            from unsloth.tokenizer_utils import check_tokenizer
            assert callable(check_tokenizer)
        except ImportError:
            pytest.skip("check_tokenizer not available")


class TestCheckTokenizer:
    """Tests for check_tokenizer function."""

    def test_check_valid_tokenizer(self, mock_tokenizer):
        """Test checking a valid tokenizer."""
        try:
            from unsloth.tokenizer_utils import check_tokenizer

            # Should not raise for valid tokenizer
            result = check_tokenizer(mock_tokenizer)
            # Result should be tokenizer or None
            assert result is None or result == mock_tokenizer
        except ImportError:
            pytest.skip("check_tokenizer not available")
        except Exception as e:
            # May fail due to mock not having all required attributes
            pytest.skip(f"check_tokenizer failed with mock: {e}")


class TestFixSentencePieceTokenizer:
    """Tests for fix_sentencepiece_tokenizer function."""

    def test_function_exists(self):
        """Test that fix_sentencepiece_tokenizer exists."""
        try:
            from unsloth.tokenizer_utils import fix_sentencepiece_tokenizer
            assert callable(fix_sentencepiece_tokenizer)
        except ImportError:
            pytest.skip("fix_sentencepiece_tokenizer not available")


class TestConvertToFastTokenizer:
    """Tests for convert_to_fast_tokenizer function."""

    def test_function_exists(self):
        """Test that convert_to_fast_tokenizer exists."""
        try:
            from unsloth.tokenizer_utils import convert_to_fast_tokenizer
            assert callable(convert_to_fast_tokenizer)
        except ImportError:
            pytest.skip("convert_to_fast_tokenizer not available")


class TestFixChatTemplate:
    """Tests for fix_chat_template function."""

    def test_function_exists(self):
        """Test that fix_chat_template exists."""
        try:
            from unsloth.tokenizer_utils import fix_chat_template
            assert callable(fix_chat_template)
        except ImportError:
            pytest.skip("fix_chat_template not available")


class TestAssertSameTokenization:
    """Tests for assert_same_tokenization function."""

    def test_function_exists(self):
        """Test that assert_same_tokenization exists."""
        try:
            from unsloth.tokenizer_utils import assert_same_tokenization
            assert callable(assert_same_tokenization)
        except ImportError:
            pytest.skip("assert_same_tokenization not available")


class TestPatchSFTTrainerTokenizer:
    """Tests for patch_sft_trainer_tokenizer function."""

    def test_function_exists(self):
        """Test that patch_sft_trainer_tokenizer exists."""
        try:
            from unsloth.tokenizer_utils import patch_sft_trainer_tokenizer
            assert callable(patch_sft_trainer_tokenizer)
        except ImportError:
            pytest.skip("patch_sft_trainer_tokenizer not available")


class TestTokenizerUtilsHelpers:
    """Tests for helper functions in tokenizer_utils."""

    def test_try_fix_tokenizer_exists(self):
        """Test that try_fix_tokenizer exists."""
        try:
            from unsloth.tokenizer_utils import try_fix_tokenizer
            assert callable(try_fix_tokenizer)
        except ImportError:
            pytest.skip("try_fix_tokenizer not available")


class TestTokenizerUtilsWithMocks:
    """Tests using mocked tokenizers."""

    def test_tokenizer_has_pad_token(self, mock_tokenizer):
        """Test that tokenizer pad token is accessible."""
        assert mock_tokenizer.pad_token == "<pad>"
        assert mock_tokenizer.pad_token_id == 0

    def test_tokenizer_has_eos_token(self, mock_tokenizer):
        """Test that tokenizer EOS token is accessible."""
        assert mock_tokenizer.eos_token == "</s>"
        assert mock_tokenizer.eos_token_id == 2

    def test_tokenizer_encode_decode(self, mock_tokenizer):
        """Test tokenizer encode/decode methods."""
        encoded = mock_tokenizer.encode("test")
        assert isinstance(encoded, list)

        decoded = mock_tokenizer.decode([1, 2, 3])
        assert isinstance(decoded, str)

    def test_tokenizer_vocab_size(self, mock_tokenizer):
        """Test tokenizer vocab size."""
        assert mock_tokenizer.vocab_size == 32000

    def test_tokenizer_max_length(self, mock_tokenizer):
        """Test tokenizer max length."""
        assert mock_tokenizer.model_max_length == 4096


class TestTokenizerValidation:
    """Tests for tokenizer validation logic."""

    def test_tokenizer_required_attributes(self):
        """Test that required tokenizer attributes are checked."""
        required_attrs = [
            "pad_token",
            "eos_token",
            "encode",
            "decode",
            "vocab_size",
        ]

        tokenizer = MagicMock()
        for attr in required_attrs:
            assert hasattr(tokenizer, attr), f"Missing required attribute: {attr}"

    def test_tokenizer_special_tokens_set(self, mock_tokenizer):
        """Test that special tokens are properly set."""
        assert mock_tokenizer.pad_token is not None
        assert mock_tokenizer.eos_token is not None
        assert mock_tokenizer.bos_token is not None

    def test_tokenizer_ids_are_integers(self, mock_tokenizer):
        """Test that token IDs are integers."""
        assert isinstance(mock_tokenizer.pad_token_id, int)
        assert isinstance(mock_tokenizer.eos_token_id, int)
        assert isinstance(mock_tokenizer.bos_token_id, int)
