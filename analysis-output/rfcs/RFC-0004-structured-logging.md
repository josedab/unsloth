# RFC-0004: Implement Structured Logging

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 3 person-days
**Priority:** P1 (Quick Win)

---

## Summary

Replace ~550 `print()` statements with structured logging using Python's logging module, enabling better debugging, filtering, and production monitoring.

---

## Motivation

Unsloth currently uses `print()` for all user communication. This prevents:
- Log level filtering
- Structured log analysis
- Integration with monitoring systems
- Correlation of related log messages

### Current State

```python
# Typical pattern throughout codebase
print("Unsloth: Loading model...")
print(f"Unsloth: Max sequence length = {max_seq_length}")
```

### Desired State

```python
logger.info("Loading model", extra={"model": model_name, "seq_length": max_seq_length})
```

---

## Detailed Design

### Logger Configuration

```python
# unsloth/logging.py
import logging
import sys
from typing import Optional

# Create Unsloth logger
logger = logging.getLogger("unsloth")

def configure_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream: Optional[object] = None,
) -> None:
    """Configure Unsloth logging."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(format))

    logger.addHandler(handler)
    logger.setLevel(level)

# Auto-configure on import
configure_logging()
```

### Usage Pattern

```python
# In any module
from unsloth.logging import logger

def from_pretrained(model_name, **kwargs):
    logger.info(f"Loading model: {model_name}")

    try:
        model = load_model(model_name)
        logger.debug(f"Model loaded successfully: {model.config}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    return model
```

### Log Levels

| Level | Use Case | Example |
|-------|----------|---------|
| DEBUG | Internal details | "Cache hit for buffer X" |
| INFO | Normal operation | "Loading model..." |
| WARNING | Recoverable issues | "Flash attention not available" |
| ERROR | Failures | "Model loading failed" |

---

## Implementation Plan

### Phase 1: Infrastructure (Day 1)
1. Create `unsloth/logging.py`
2. Configure default logger
3. Add environment variable control

### Phase 2: Replace Print Statements (Days 2-3)
1. Replace prints in high-priority files:
   - `models/loader.py`
   - `models/_utils.py`
   - `save.py`
2. Add appropriate log levels

### Phase 3: Testing (Day 3)
1. Verify log output
2. Test filtering
3. Update documentation

---

## Success Criteria

- [ ] 0 `print()` statements for user messaging
- [ ] All messages use appropriate log levels
- [ ] Log filtering works via environment variable
- [ ] Structured data in log messages

---

*Next: [RFC-0005: Comprehensive Documentation](./RFC-0005-documentation.md)*
