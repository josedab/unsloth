# RFC-0005: Comprehensive Documentation

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 15 person-days
**Priority:** P1 (Strategic)

---

## Summary

Increase docstring coverage from 33% to 80%+ and create comprehensive API documentation with examples for all public functions.

---

## Motivation

Only 33% of functions have docstrings. The registry module has only 2.2% coverage. This makes the codebase difficult to understand and contribute to.

### Impact

- Contributors must read implementation to understand API
- IDE help is unavailable
- No generated documentation possible
- Longer onboarding time

---

## Detailed Design

### Docstring Standard

Use Google-style docstrings:

```python
def from_pretrained(
    model_name: str,
    max_seq_length: Optional[int] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model with Unsloth optimizations.

    This function loads a HuggingFace model and applies Unsloth
    optimizations including Flash Attention, optimized kernels,
    and memory management.

    Args:
        model_name: HuggingFace model ID or local path.
        max_seq_length: Maximum sequence length. If None, uses model default.

    Returns:
        Tuple of (model, tokenizer) ready for training.

    Raises:
        ValueError: If model is not supported.
        RuntimeError: If loading fails.

    Example:
        >>> from unsloth import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained(
        ...     "unsloth/llama-3-8b-bnb-4bit",
        ...     max_seq_length=2048,
        ... )

    Note:
        Import unsloth before transformers to ensure patches apply.
    """
```

### Priority Modules

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `models/loader.py` | 15% | 100% | P0 |
| `save.py` | 10% | 100% | P0 |
| `registry/` | 2.2% | 100% | P0 |
| `trainer.py` | 20% | 100% | P1 |
| `kernels/` | 15.7% | 80% | P2 |

---

## Implementation Plan

### Phase 1: Public API (Days 1-5)
Document user-facing functions:
- FastLanguageModel
- UnslothTrainer
- Save functions

### Phase 2: Registry (Days 6-8)
Document model registration and lookup.

### Phase 3: Kernels (Days 9-12)
Document optimization kernels.

### Phase 4: Examples (Days 13-15)
Add runnable examples to each docstring.

---

## Success Criteria

- [ ] 80%+ docstring coverage
- [ ] All public functions have examples
- [ ] Generated documentation builds
- [ ] Docstrings include types

---

*Next: [RFC-0006: Automated CI Testing Pipeline](./RFC-0006-ci-testing.md)*
