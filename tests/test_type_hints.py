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

"""
Type hints validation tests for the Unsloth library.

This module contains tests that verify the type hints are correct and
can be validated by static type checkers like mypy.

Run with:
    mypy tests/test_type_hints.py

Or run the test directly:
    python -m pytest tests/test_type_hints.py -v
"""

import unittest
from typing import get_type_hints
from pathlib import Path


class TestTypeHintsExist(unittest.TestCase):
    """Test that type hints are properly defined on public API functions."""

    def test_types_module_exists(self):
        """Test that the types module can be imported."""
        from unsloth.types import (
            ModelName,
            DType,
            DeviceMap,
            ModelTokenizerTuple,
            QuantizationMethod,
            SaveMethod,
            BiasType,
            GradientCheckpointing,
            ChatTemplateType,
            MappingType,
        )
        # Verify type aliases exist
        self.assertIsNotNone(ModelName)
        self.assertIsNotNone(DType)
        self.assertIsNotNone(DeviceMap)
        self.assertIsNotNone(ModelTokenizerTuple)
        self.assertIsNotNone(QuantizationMethod)

    def test_py_typed_marker_exists(self):
        """Test that py.typed marker file exists for PEP 561 compliance."""
        import unsloth
        package_dir = Path(unsloth.__file__).parent
        py_typed = package_dir / "py.typed"
        self.assertTrue(py_typed.exists(), "py.typed marker file should exist")

    def test_fast_language_model_type_hints(self):
        """Test that FastLanguageModel.from_pretrained has type hints."""
        from unsloth import FastLanguageModel

        # Get type hints for from_pretrained
        hints = get_type_hints(FastLanguageModel.from_pretrained)

        # Check that return type is specified
        self.assertIn('return', hints, "from_pretrained should have return type hint")

        # Check key parameters have type hints
        self.assertIn('model_name', hints)
        self.assertIn('max_seq_length', hints)
        self.assertIn('dtype', hints)
        self.assertIn('load_in_4bit', hints)

    def test_unsloth_trainer_type_hints(self):
        """Test that UnslothTrainer has type hints."""
        from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments

        # Check that UnslothTrainingArguments.__init__ has hints
        hints = get_type_hints(UnslothTrainingArguments.__init__)
        self.assertIn('embedding_learning_rate', hints)

        # Check that UnslothTrainer.create_optimizer has hints
        hints = get_type_hints(UnslothTrainer.create_optimizer)
        self.assertIn('return', hints)

    def test_save_functions_type_hints(self):
        """Test that save functions have type hints."""
        from unsloth.save import (
            save_to_gguf,
            unsloth_save_pretrained_merged,
            unsloth_push_to_hub_merged,
        )

        # Check save_to_gguf
        hints = get_type_hints(save_to_gguf)
        self.assertIn('return', hints)
        self.assertIn('model_name', hints)
        self.assertIn('quantization_method', hints)

        # Check unsloth_save_pretrained_merged
        hints = get_type_hints(unsloth_save_pretrained_merged)
        self.assertIn('return', hints)
        self.assertIn('save_directory', hints)
        self.assertIn('tokenizer', hints)

        # Check unsloth_push_to_hub_merged
        hints = get_type_hints(unsloth_push_to_hub_merged)
        self.assertIn('return', hints)
        self.assertIn('repo_id', hints)

    def test_chat_template_type_hints(self):
        """Test that chat template functions have type hints."""
        from unsloth.chat_templates import get_chat_template, apply_chat_template

        # Check get_chat_template
        hints = get_type_hints(get_chat_template)
        self.assertIn('return', hints)
        self.assertIn('tokenizer', hints)
        self.assertIn('chat_template', hints)

        # Check apply_chat_template
        hints = get_type_hints(apply_chat_template)
        self.assertIn('return', hints)
        self.assertIn('dataset', hints)

    def test_tokenizer_utils_type_hints(self):
        """Test that tokenizer utils have type hints."""
        from unsloth.tokenizer_utils import (
            load_correct_tokenizer,
            fix_sentencepiece_gguf,
        )

        # Check load_correct_tokenizer
        hints = get_type_hints(load_correct_tokenizer)
        self.assertIn('return', hints)
        self.assertIn('tokenizer_name', hints)
        self.assertIn('model_max_length', hints)

        # Check fix_sentencepiece_gguf
        hints = get_type_hints(fix_sentencepiece_gguf)
        self.assertIn('return', hints)
        self.assertIn('saved_location', hints)


class TestTypeAnnotationExamples(unittest.TestCase):
    """Test example code that should type-check correctly."""

    def test_model_name_type_accepts_string(self):
        """Test that ModelName accepts str."""
        from unsloth.types import ModelName
        from typing import get_origin, get_args, Union

        # ModelName should be Union[str, Path]
        origin = get_origin(ModelName)
        args = get_args(ModelName)

        self.assertEqual(origin, Union)
        self.assertIn(str, args)
        self.assertIn(Path, args)

    def test_dtype_literal_values(self):
        """Test that DType accepts the correct literal values."""
        from unsloth.types import DType
        from typing import get_origin, get_args, Union, Literal

        # DType should be Optional[Literal[...]]
        origin = get_origin(DType)
        args = get_args(DType)

        # Should allow None
        self.assertIn(type(None), args)

    def test_quantization_method_literal(self):
        """Test that QuantizationMethod is a Literal type."""
        from unsloth.types import QuantizationMethod
        from typing import get_origin, Literal

        # QuantizationMethod should be a Literal
        origin = get_origin(QuantizationMethod)
        self.assertEqual(origin, Literal)


if __name__ == "__main__":
    unittest.main()
