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
Tests for Unsloth structured logging functionality.
"""

import io
import json
import logging
import os
import sys
import unittest
from unittest import mock


class TestLoggingModule(unittest.TestCase):
    """Test cases for unsloth.logging module."""

    def setUp(self):
        """Reset logger state before each test."""
        # Import fresh to reset state
        from unsloth import logging as unsloth_logging

        self.unsloth_logging = unsloth_logging
        self.logger = unsloth_logging.logger

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Reset to default state
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        """Clean up after each test."""
        # Remove any file handlers
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self.logger.removeHandler(handler)

    def test_logger_creation(self):
        """Test that logger is created with correct name."""
        self.assertEqual(self.logger.name, "unsloth")
        self.assertFalse(self.logger.propagate)

    def test_configure_logging_default(self):
        """Test default logging configuration."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)

        self.logger.info("Test message")

        output = stream.getvalue()
        self.assertIn("Test message", output)
        self.assertIn("unsloth", output)
        self.assertIn("INFO", output)

    def test_configure_logging_debug_level(self):
        """Test configuring DEBUG log level."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(level=logging.DEBUG, stream=stream)

        self.logger.debug("Debug message")
        self.logger.info("Info message")

        output = stream.getvalue()
        self.assertIn("Debug message", output)
        self.assertIn("Info message", output)

    def test_configure_logging_warning_level(self):
        """Test configuring WARNING log level filters lower levels."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(level=logging.WARNING, stream=stream)

        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")

        output = stream.getvalue()
        self.assertNotIn("Debug message", output)
        self.assertNotIn("Info message", output)
        self.assertIn("Warning message", output)

    def test_configure_logging_custom_format(self):
        """Test custom log format."""
        stream = io.StringIO()
        custom_format = "[%(levelname)s] %(message)s"
        self.unsloth_logging.configure_logging(format=custom_format, stream=stream)

        self.logger.info("Test message")

        output = stream.getvalue()
        self.assertIn("[INFO] Test message", output)

    def test_configure_logging_force_replaces_handlers(self):
        """Test force parameter removes existing handlers."""
        stream1 = io.StringIO()
        stream2 = io.StringIO()

        self.unsloth_logging.configure_logging(stream=stream1)
        self.logger.info("First message")

        self.unsloth_logging.configure_logging(stream=stream2, force=True)
        self.logger.info("Second message")

        # First message should only be in stream1
        self.assertIn("First message", stream1.getvalue())
        self.assertNotIn("Second message", stream1.getvalue())

        # Second message should only be in stream2
        self.assertNotIn("First message", stream2.getvalue())
        self.assertIn("Second message", stream2.getvalue())

    def test_set_log_level(self):
        """Test set_log_level function."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)

        self.unsloth_logging.set_log_level(logging.ERROR)

        self.logger.warning("Warning message")
        self.logger.error("Error message")

        output = stream.getvalue()
        self.assertNotIn("Warning message", output)
        self.assertIn("Error message", output)

    def test_log_levels(self):
        """Test all log levels work correctly."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(level=logging.DEBUG, stream=stream)

        self.logger.debug("Debug level")
        self.logger.info("Info level")
        self.logger.warning("Warning level")
        self.logger.error("Error level")
        self.logger.critical("Critical level")

        output = stream.getvalue()
        self.assertIn("DEBUG", output)
        self.assertIn("INFO", output)
        self.assertIn("WARNING", output)
        self.assertIn("ERROR", output)
        self.assertIn("CRITICAL", output)


class TestEnvironmentVariableControl(unittest.TestCase):
    """Test environment variable control of logging."""

    def setUp(self):
        """Save original environment and reset logger."""
        self.original_env = os.environ.copy()

        # Clean environment
        for key in ["UNSLOTH_LOG_LEVEL", "UNSLOTH_ENABLE_LOGGING"]:
            os.environ.pop(key, None)

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_get_log_level_from_env_debug(self):
        """Test UNSLOTH_LOG_LEVEL=DEBUG."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "DEBUG"

        # Reimport to get fresh state
        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.DEBUG)

    def test_get_log_level_from_env_info(self):
        """Test UNSLOTH_LOG_LEVEL=INFO."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "INFO"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.INFO)

    def test_get_log_level_from_env_warning(self):
        """Test UNSLOTH_LOG_LEVEL=WARNING."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "WARNING"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.WARNING)

    def test_get_log_level_from_env_error(self):
        """Test UNSLOTH_LOG_LEVEL=ERROR."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "ERROR"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.ERROR)

    def test_get_log_level_from_env_critical(self):
        """Test UNSLOTH_LOG_LEVEL=CRITICAL."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "CRITICAL"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.CRITICAL)

    def test_get_log_level_from_env_case_insensitive(self):
        """Test that log level is case insensitive."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "debug"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.DEBUG)

    def test_legacy_enable_logging_disabled(self):
        """Test legacy UNSLOTH_ENABLE_LOGGING=0 sets WARNING level."""
        os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.WARNING)

    def test_legacy_enable_logging_enabled(self):
        """Test legacy UNSLOTH_ENABLE_LOGGING=1 sets INFO level."""
        os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.INFO)

    def test_new_env_var_takes_precedence(self):
        """Test UNSLOTH_LOG_LEVEL takes precedence over UNSLOTH_ENABLE_LOGGING."""
        os.environ["UNSLOTH_LOG_LEVEL"] = "DEBUG"
        os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"

        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.DEBUG)

    def test_default_level_is_info(self):
        """Test default level is INFO when no env vars set."""
        from unsloth.logging import get_log_level_from_env

        level = get_log_level_from_env()
        self.assertEqual(level, logging.INFO)


class TestFileHandler(unittest.TestCase):
    """Test file handler functionality."""

    def setUp(self):
        """Reset logger state."""
        from unsloth import logging as unsloth_logging

        self.unsloth_logging = unsloth_logging
        self.logger = unsloth_logging.logger

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.test_file = "/tmp/test_unsloth_logging.log"

    def tearDown(self):
        """Clean up handlers and test file."""
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self.logger.removeHandler(handler)

        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_add_file_handler(self):
        """Test adding a file handler."""
        # First configure with a stream handler
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)

        # Add file handler
        handler = self.unsloth_logging.add_file_handler(self.test_file)

        self.logger.info("File test message")
        handler.flush()

        # Check file contains message
        with open(self.test_file, "r") as f:
            content = f.read()

        self.assertIn("File test message", content)

    def test_file_handler_separate_level(self):
        """Test file handler with separate log level."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(level=logging.DEBUG, stream=stream)

        # File handler only logs ERROR and above
        handler = self.unsloth_logging.add_file_handler(
            self.test_file,
            level=logging.ERROR
        )

        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.error("Error message")
        handler.flush()

        # Stream should have all messages
        stream_output = stream.getvalue()
        self.assertIn("Debug message", stream_output)
        self.assertIn("Info message", stream_output)
        self.assertIn("Error message", stream_output)

        # File should only have error
        with open(self.test_file, "r") as f:
            file_content = f.read()

        self.assertNotIn("Debug message", file_content)
        self.assertNotIn("Info message", file_content)
        self.assertIn("Error message", file_content)


class TestJSONLogging(unittest.TestCase):
    """Test JSON logging functionality."""

    def setUp(self):
        """Reset logger state."""
        from unsloth import logging as unsloth_logging

        self.unsloth_logging = unsloth_logging
        self.logger = unsloth_logging.logger

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def tearDown(self):
        """Clean up handlers."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def test_enable_json_logging(self):
        """Test JSON formatted logging output."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)
        self.unsloth_logging.enable_json_logging()

        self.logger.info("JSON test message")

        output = stream.getvalue().strip()

        # Parse as JSON
        log_data = json.loads(output)

        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["logger"], "unsloth")
        self.assertEqual(log_data["message"], "JSON test message")
        self.assertIn("timestamp", log_data)

    def test_json_logging_with_extra_fields(self):
        """Test JSON logging includes extra fields."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)
        self.unsloth_logging.enable_json_logging()

        self.logger.info(
            "Model loaded",
            extra={"model": "llama-2-7b", "seq_length": 2048}
        )

        output = stream.getvalue().strip()
        log_data = json.loads(output)

        self.assertEqual(log_data["message"], "Model loaded")
        self.assertEqual(log_data["model"], "llama-2-7b")
        self.assertEqual(log_data["seq_length"], 2048)


class TestLoggerImport(unittest.TestCase):
    """Test that logger can be imported correctly."""

    def test_import_from_unsloth_logging(self):
        """Test importing logger from unsloth.logging."""
        from unsloth.logging import logger

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "unsloth")

    def test_import_configure_logging(self):
        """Test importing configure_logging function."""
        from unsloth.logging import configure_logging

        self.assertTrue(callable(configure_logging))

    def test_import_all_exports(self):
        """Test all expected exports are available."""
        from unsloth.logging import (
            logger,
            configure_logging,
            get_log_level_from_env,
        )

        self.assertIsNotNone(logger)
        self.assertTrue(callable(configure_logging))
        self.assertTrue(callable(get_log_level_from_env))


class TestLogMessageFormatting(unittest.TestCase):
    """Test log message formatting."""

    def setUp(self):
        """Reset logger state."""
        from unsloth import logging as unsloth_logging

        self.unsloth_logging = unsloth_logging
        self.logger = unsloth_logging.logger

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def tearDown(self):
        """Clean up handlers."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def test_unsloth_format(self):
        """Test default Unsloth format."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(
            format=self.unsloth_logging.UNSLOTH_FORMAT,
            stream=stream
        )

        self.logger.info("Test message")

        output = stream.getvalue()
        self.assertIn("Unsloth: Test message", output)

    def test_simple_format(self):
        """Test simple format."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(
            format=self.unsloth_logging.SIMPLE_FORMAT,
            stream=stream
        )

        self.logger.info("Test message")

        output = stream.getvalue()
        self.assertIn("[INFO] Test message", output)

    def test_fstring_messages(self):
        """Test f-string log messages work correctly."""
        stream = io.StringIO()
        self.unsloth_logging.configure_logging(stream=stream)

        model_name = "llama-2-7b"
        seq_length = 2048
        self.logger.info(f"Loading model: {model_name}, seq_length: {seq_length}")

        output = stream.getvalue()
        self.assertIn("Loading model: llama-2-7b", output)
        self.assertIn("seq_length: 2048", output)


if __name__ == "__main__":
    unittest.main()
