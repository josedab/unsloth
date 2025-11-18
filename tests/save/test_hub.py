# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for unsloth.save.hub module."""

import pytest
from unittest.mock import MagicMock, patch

from unsloth.save.hub import (
    determine_username,
    MODEL_CARD,
)
from unsloth.save.base import HubError


class TestDetermineUsername:
    """Tests for determine_username function."""

    def test_username_from_repo_id(self):
        """Test extracting username from repo_id with slash."""
        result, username = determine_username("myuser/mymodel", None, None)
        assert result == "myuser/mymodel"
        assert username == "myuser"

    def test_strips_leading_chars(self):
        """Test that leading ./ is stripped."""
        result, username = determine_username("./myuser/mymodel", None, None)
        assert result == "myuser/mymodel"
        assert username == "myuser"

    @patch("unsloth.save.hub.whoami")
    def test_username_from_token(self, mock_whoami):
        """Test getting username from token when not in path."""
        mock_whoami.return_value = {"name": "testuser"}
        result, username = determine_username("mymodel", None, "test_token")
        assert result == "testuser/mymodel"
        assert username == "testuser"

    @patch("unsloth.save.hub.whoami")
    def test_old_username_preserved(self, mock_whoami):
        """Test that old_username is preserved when different."""
        mock_whoami.return_value = {"name": "newuser"}
        result, username = determine_username("mymodel", "olduser", "test_token")
        assert username == "olduser"

    @patch("unsloth.save.hub.whoami")
    def test_invalid_directory_raises_error(self, mock_whoami):
        """Test that invalid directory raises HubError."""
        mock_whoami.side_effect = Exception("Auth failed")
        with pytest.raises(HubError) as exc_info:
            determine_username("mymodel", None, "bad_token")
        assert "not a valid HuggingFace directory" in str(exc_info.value)


class TestModelCard:
    """Tests for MODEL_CARD template."""

    def test_model_card_format(self):
        """Test that MODEL_CARD can be formatted."""
        result = MODEL_CARD.format(
            base_model="meta-llama/Llama-2-7b",
            model_type="llama",
            extra="custom_tag",
            method="finetuned",
            username="testuser",
        )
        assert "meta-llama/Llama-2-7b" in result
        assert "testuser" in result
        assert "unsloth" in result
        assert "Unsloth" in result

    def test_model_card_contains_required_fields(self):
        """Test that MODEL_CARD contains required metadata."""
        assert "base_model" in MODEL_CARD
        assert "tags" in MODEL_CARD
        assert "license" in MODEL_CARD
        assert "unsloth" in MODEL_CARD
