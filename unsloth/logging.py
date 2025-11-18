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
Structured logging for Unsloth.

This module provides a centralized logging configuration for the Unsloth package.
Users can control logging behavior via environment variables or programmatically.

Example usage:
    from unsloth.logging import logger, configure_logging

    # Use default configuration
    logger.info("Loading model", extra={"model": "llama-2-7b"})

    # Configure logging programmatically
    import logging
    configure_logging(level=logging.DEBUG)

    # Environment variable control:
    # export UNSLOTH_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import logging
import os
import sys
from typing import Optional, TextIO

__all__ = ["logger", "configure_logging", "get_log_level_from_env"]

# Create Unsloth logger
logger = logging.getLogger("unsloth")

# Prevent propagation to root logger to avoid duplicate messages
logger.propagate = False

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "[%(levelname)s] %(message)s"
UNSLOTH_FORMAT = "Unsloth: %(message)s"


def get_log_level_from_env() -> int:
    """
    Get log level from environment variable.

    Supports:
        - UNSLOTH_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
        - UNSLOTH_ENABLE_LOGGING: Legacy support (0 = WARNING, 1 = INFO)

    Returns:
        logging level constant (e.g., logging.INFO)
    """
    # Check new environment variable first
    level_str = os.environ.get("UNSLOTH_LOG_LEVEL", "").upper()
    if level_str:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level_str, logging.INFO)

    # Legacy support for UNSLOTH_ENABLE_LOGGING
    enable_logging = os.environ.get("UNSLOTH_ENABLE_LOGGING", "1")
    if enable_logging == "0":
        return logging.WARNING

    return logging.INFO


def configure_logging(
    level: Optional[int] = None,
    format: str = UNSLOTH_FORMAT,
    stream: Optional[TextIO] = None,
    force: bool = False,
) -> None:
    """
    Configure Unsloth logging.

    Args:
        level: Logging level (e.g., logging.INFO). If None, uses environment variable.
        format: Log message format string.
        stream: Output stream (defaults to sys.stderr).
        force: If True, removes existing handlers before adding new one.

    Example:
        import logging
        from unsloth.logging import configure_logging

        # Verbose debugging
        configure_logging(level=logging.DEBUG)

        # Quiet mode (warnings only)
        configure_logging(level=logging.WARNING)

        # Custom format
        configure_logging(format="[%(levelname)s] %(message)s")
    """
    if force and logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(format))
        logger.addHandler(handler)
    else:
        # Update existing handler's formatter
        for handler in logger.handlers:
            handler.setFormatter(logging.Formatter(format))

    # Set level from argument or environment
    if level is None:
        level = get_log_level_from_env()
    logger.setLevel(level)


def set_log_level(level: int) -> None:
    """
    Set the log level for the Unsloth logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        import logging
        from unsloth.logging import set_log_level

        set_log_level(logging.DEBUG)
    """
    logger.setLevel(level)


def add_file_handler(
    filepath: str,
    level: Optional[int] = None,
    format: str = DEFAULT_FORMAT,
) -> logging.FileHandler:
    """
    Add a file handler to the Unsloth logger.

    Args:
        filepath: Path to the log file.
        level: Optional separate level for file logging.
        format: Log message format string.

    Returns:
        The created FileHandler.

    Example:
        from unsloth.logging import add_file_handler

        add_file_handler("/tmp/unsloth.log")
    """
    handler = logging.FileHandler(filepath)
    handler.setFormatter(logging.Formatter(format))
    if level is not None:
        handler.setLevel(level)
    logger.addHandler(handler)
    return handler


def enable_json_logging() -> None:
    """
    Enable JSON-formatted logging for structured log aggregation.

    This is useful for integration with log aggregation systems like
    ELK stack, Splunk, or cloud logging services.

    Example:
        from unsloth.logging import enable_json_logging

        enable_json_logging()
        # Logs will now be in JSON format
    """
    import json
    from datetime import datetime

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add extra fields if present
            if hasattr(record, "__dict__"):
                for key, value in record.__dict__.items():
                    if key not in (
                        "name", "msg", "args", "created", "filename", "funcName",
                        "levelname", "levelno", "lineno", "module", "msecs",
                        "pathname", "process", "processName", "relativeCreated",
                        "stack_info", "thread", "threadName", "exc_info", "exc_text",
                        "message",
                    ):
                        log_data[key] = value

            return json.dumps(log_data)

    for handler in logger.handlers:
        handler.setFormatter(JsonFormatter())


# Auto-configure on import using environment variables
configure_logging()
