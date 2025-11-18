# Unsloth Structured Logging

Unsloth uses Python's standard logging module for structured, configurable logging output.

## Quick Start

By default, Unsloth logs INFO-level messages to stderr. No configuration is needed for basic usage.

```python
from unsloth import FastLanguageModel

# Normal usage - logging works automatically
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## Controlling Log Levels

### Environment Variables

Set the log level via environment variable:

```bash
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
export UNSLOTH_LOG_LEVEL=DEBUG

# Quiet mode (warnings only)
export UNSLOTH_LOG_LEVEL=WARNING

# Disable most logging
export UNSLOTH_LOG_LEVEL=CRITICAL
```

### Programmatic Control

Configure logging in your Python code:

```python
import logging
from unsloth.logging import configure_logging, set_log_level

# Verbose debugging
configure_logging(level=logging.DEBUG)

# Quiet mode
configure_logging(level=logging.WARNING)

# Custom format
configure_logging(format="[%(levelname)s] %(message)s")

# Or just change the level
set_log_level(logging.DEBUG)
```

## Log Levels

| Level | Use Case | Example Messages |
|-------|----------|------------------|
| DEBUG | Internal details, verbose output | "Cache hit for buffer X" |
| INFO | Normal operation | "Saving model...", "Done." |
| WARNING | Recoverable issues | "Flash attention not available" |
| ERROR | Failures | "Model loading failed" |
| CRITICAL | Severe errors | System-breaking issues |

## Advanced Configuration

### File Logging

Log to a file in addition to stderr:

```python
from unsloth.logging import configure_logging, add_file_handler

configure_logging()
add_file_handler("/path/to/unsloth.log")
```

### Separate File Log Level

Keep console quiet but log everything to file:

```python
import logging
from unsloth.logging import configure_logging, add_file_handler

# Console only shows warnings
configure_logging(level=logging.WARNING)

# File captures everything
add_file_handler("/path/to/unsloth.log", level=logging.DEBUG)
```

### JSON Logging

For log aggregation systems (ELK, Splunk, etc.):

```python
from unsloth.logging import configure_logging, enable_json_logging

configure_logging()
enable_json_logging()
```

Output format:
```json
{
  "timestamp": "2025-11-18T12:00:00.000Z",
  "level": "INFO",
  "logger": "unsloth",
  "message": "Saving model..."
}
```

### Using Extra Fields

Add structured data to log messages:

```python
from unsloth.logging import logger

logger.info("Model loaded", extra={
    "model": "llama-2-7b",
    "seq_length": 2048,
    "load_in_4bit": True
})
```

## Legacy Support

The old `UNSLOTH_ENABLE_LOGGING` environment variable is still supported:

```bash
# Equivalent to INFO level
export UNSLOTH_ENABLE_LOGGING=1

# Equivalent to WARNING level
export UNSLOTH_ENABLE_LOGGING=0
```

Note: `UNSLOTH_LOG_LEVEL` takes precedence if both are set.

## Integration with Python Logging

The Unsloth logger is a standard Python logger named "unsloth":

```python
import logging

# Get the Unsloth logger
unsloth_logger = logging.getLogger("unsloth")

# Integrate with your application's logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
```

## Filtering Unsloth Logs

To filter out Unsloth logs from your application:

```python
import logging

# Only show warnings and above from Unsloth
logging.getLogger("unsloth").setLevel(logging.WARNING)
```

## Migration from Print Statements

If you were parsing stdout for Unsloth messages, note that:

1. Logs now go to stderr by default (stdout unchanged)
2. Message format includes timestamps and log levels
3. Use `UNSLOTH_LOG_LEVEL=CRITICAL` to suppress most output

For parsing structured logs, consider using JSON logging mode.

## Troubleshooting

### No Log Output

Check that the log level isn't set too high:

```python
from unsloth.logging import logger
print(f"Current level: {logger.level}")  # 10=DEBUG, 20=INFO, 30=WARNING, etc.
```

### Duplicate Messages

Ensure `configure_logging` isn't called multiple times without `force=True`:

```python
from unsloth.logging import configure_logging

# This replaces previous configuration
configure_logging(level=logging.DEBUG, force=True)
```

### Performance

Logging has minimal performance overhead. For training loops where performance is critical, you can temporarily increase the log level:

```python
import logging
from unsloth.logging import set_log_level

# Quiet during training
set_log_level(logging.WARNING)

# ... training code ...

# Restore after training
set_log_level(logging.INFO)
```
