# RFC-0005: Comprehensive Documentation

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 15 person-days
**Priority:** P1 (Strategic)

---

## Summary

Increase docstring coverage from 33% to 80%+ and create comprehensive API documentation with examples for all public functions, enabling better onboarding, IDE support, and automated documentation generation.

---

## Motivation

Only 33% of functions (206 of 625) have docstrings. The registry module has only 2.2% coverage (1 of 45 functions). This makes the codebase difficult to understand, contribute to, and use effectively.

### Current Problem

```python
# From unsloth/models/loader.py - no docstring
def from_pretrained(model_name, max_seq_length=None, dtype=None, load_in_4bit=True):
    ...
```

Users must:
- Read implementation to understand parameters
- Guess at valid values
- Find examples elsewhere
- Learn by trial and error

### Impact

| Issue | Effect |
|-------|--------|
| No IDE help | Reduced developer productivity |
| No generated docs | Manual documentation required |
| Slower onboarding | Contributors must read implementation |
| More support requests | Users can't self-serve |

### Documentation Coverage by Module

| Module | Functions | Documented | Coverage |
|--------|-----------|------------|----------|
| `registry/` | 45 | 1 | 2.2% |
| `models/` | 209 | 29 | 13.9% |
| `kernels/` | 64 | 10 | 15.7% |
| `chat_templates.py` | 54 | 42 | 77.8% |
| **Overall** | **625** | **206** | **33%** |

---

## Detailed Design

### Docstring Standard: Google Style

All docstrings will follow Google style for consistency:

```python
def from_pretrained(
    model_name: str,
    max_seq_length: Optional[int] = None,
    dtype: Optional[str] = None,
    load_in_4bit: bool = True,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model with Unsloth optimizations.

    This function loads a HuggingFace model and applies Unsloth
    optimizations including Flash Attention, optimized Triton kernels,
    and memory-efficient gradient checkpointing.

    Args:
        model_name: HuggingFace model ID (e.g., "unsloth/llama-3-8b-bnb-4bit")
            or local path to model directory.
        max_seq_length: Maximum sequence length for training. If None,
            uses the model's default from config.json. Longer sequences
            use more memory.
        dtype: Data type for model weights. Options: "float16", "bfloat16",
            "float32". If None, auto-detects based on GPU capability.
        load_in_4bit: Whether to load model in 4-bit quantization.
            Reduces memory by ~75% with minimal quality loss.
        **kwargs: Additional arguments passed to
            AutoModelForCausalLM.from_pretrained().

    Returns:
        Tuple of (model, tokenizer) ready for training or inference.

    Raises:
        ValueError: If model_name is not a supported architecture.
        RuntimeError: If model loading fails due to memory or network issues.

    Example:
        Basic usage:

        >>> from unsloth import FastLanguageModel
        >>> model, tokenizer = FastLanguageModel.from_pretrained(
        ...     "unsloth/llama-3-8b-bnb-4bit",
        ...     max_seq_length=2048,
        ... )

        With custom dtype:

        >>> model, tokenizer = FastLanguageModel.from_pretrained(
        ...     "meta-llama/Llama-3-8B",
        ...     dtype="bfloat16",
        ...     load_in_4bit=False,
        ... )

    Note:
        Import unsloth before transformers to ensure patches are applied:

        >>> import unsloth  # Must be first!
        >>> from transformers import ...

    See Also:
        - :meth:`get_peft_model`: Apply LoRA after loading
        - :meth:`for_inference`: Optimize for inference
    """
```

### Required Sections

| Section | Required | Description |
|---------|----------|-------------|
| Summary | Yes | One-line description |
| Description | For complex funcs | Extended explanation |
| Args | Yes | All parameters |
| Returns | Yes | Return value(s) |
| Raises | If applicable | Exceptions raised |
| Example | Yes for public API | Runnable code |
| Note | If applicable | Important caveats |
| See Also | If applicable | Related functions |

### Priority Modules

#### Priority 0: Public API (must have 100%)

| File | Functions | Target |
|------|-----------|--------|
| `models/loader.py` | 12 | 100% |
| `save.py` | 15 | 100% |
| `trainer.py` | 8 | 100% |
| Registry public functions | 10 | 100% |

#### Priority 1: Core Internals (80%)

| File | Functions | Target |
|------|-----------|--------|
| `models/llama.py` | 49 | 80% |
| `models/_utils.py` | 45 | 80% |
| `tokenizer_utils.py` | 32 | 80% |

#### Priority 2: Kernels (60%)

| File | Functions | Target |
|------|-----------|--------|
| `kernels/fast_lora.py` | 15 | 60% |
| `kernels/utils.py` | 25 | 60% |
| Other kernels | 24 | 60% |

### Documentation Generation

Set up Sphinx for automatic documentation:

```python
# docs/conf.py
project = 'Unsloth'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google style support
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
```

---

## Example Usage

### IDE Integration

With docstrings, IDEs show:

```python
# User types:
FastLanguageModel.from_pretrained(

# IDE shows:
"""
Load a model with Unsloth optimizations.

Args:
    model_name: HuggingFace model ID or local path
    max_seq_length: Maximum sequence length (default: from config)
    ...
"""
```

### Generated Documentation

```bash
# Generate HTML docs
cd docs
make html

# Output in docs/_build/html/
```

### Doctest Validation

```bash
# Run examples in docstrings
python -m doctest unsloth/models/loader.py -v
```

---

## Implementation Plan

### Phase 1: Public API (Days 1-5)

**Day 1-2: Core Loading**
- `FastLanguageModel.from_pretrained()`
- `FastLanguageModel.get_peft_model()`
- `FastVisionModel.from_pretrained()`

**Day 3-4: Training**
- `UnslothTrainer` class
- `UnslothTrainingArguments`
- `unsloth_train()`

**Day 5: Saving**
- `save_to_gguf()`
- `save_pretrained_merged()`
- `push_to_hub_merged()`

### Phase 2: Registry (Days 6-8)

**Day 6: Core Registry**
- `register_model()`
- `get_model_info()`
- `search_models()`
- `ModelInfo` class

**Day 7-8: Model-Specific**
- Document each model registry file
- Add examples for model lookup

### Phase 3: Internals (Days 9-12)

**Day 9-10: Models**
- `llama.py` main functions
- `_utils.py` public utilities

**Day 11-12: Utilities**
- `tokenizer_utils.py`
- `chat_templates.py` (mostly done)

### Phase 4: Kernels (Days 13-14)

- `fast_lora.py` classes
- `utils.py` functions
- Other kernel functions

### Phase 5: Review and Generation (Day 15)

- Set up Sphinx
- Generate documentation
- Review and fix issues
- Write contribution guide for docs

### Milestones

| Day | Deliverable | Coverage |
|-----|-------------|----------|
| 5 | Public API complete | 100% public |
| 8 | Registry complete | 100% registry |
| 12 | Core internals | 80% core |
| 14 | Kernels | 60% kernels |
| 15 | Generated docs | 80%+ overall |

---

## Backwards Compatibility

### Breaking Changes

None. Docstrings are additive.

### Tools Compatibility

- Sphinx: Full support
- pdoc: Full support
- IDE hover: Full support
- doctest: Verified examples

---

## Alternatives Considered

### Alternative 1: README-only Documentation

Document only in README.md without docstrings.

**Rejected:**
- No IDE integration
- Examples can drift from code
- Harder to maintain

### Alternative 2: Separate Documentation Site Only

Write docs separately from code.

**Rejected:**
- Documentation drifts from code
- No IDE help
- Duplicate maintenance

### Alternative 3: Type Stubs with Comments

Use .pyi files with comments.

**Rejected:**
- Comments not rendered by tools
- Separate from implementation
- Less discoverable

### Alternative 4: Minimal Docstrings

Only one-line descriptions.

**Rejected:**
- No examples
- No parameter documentation
- Limited IDE help

---

## Open Questions

1. **Should we require doctest-passing examples?**
   - Ensures examples work
   - Adds CI complexity
   - Some examples need GPU

2. **Host documentation where?**
   - ReadTheDocs (free, standard)
   - GitHub Pages (integrated)
   - Self-hosted

3. **Include internal functions?**
   - Pro: Complete documentation
   - Con: More maintenance, may confuse users

4. **Translate documentation?**
   - Large international user base
   - Significant effort

---

## Success Criteria

- [ ] 100% coverage for public API
- [ ] 80% coverage for core modules
- [ ] All public functions have examples
- [ ] Sphinx documentation builds without warnings
- [ ] Examples are runnable (doctest passes where possible)
- [ ] Contribution guide for documentation

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Overall coverage | 33% | 80%+ |
| Public API coverage | ~15% | 100% |
| Registry coverage | 2.2% | 100% |
| Generated docs | No | Yes |

---

## Required Approvals

- [ ] Maintainer review of docstring standard
- [ ] Technical review of examples
- [ ] Documentation hosting decision

---

## Rollback Strategy

Docstrings can be removed without breaking functionality. Individual modules can be reverted if issues arise.

---

## References

- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- Sphinx Napoleon Extension: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
- NumPy Docstring Standard: https://numpydoc.readthedocs.io/en/latest/format.html
- Example: HuggingFace Transformers Docs

---

*Next: [RFC-0006: Automated CI Testing Pipeline](./RFC-0006-ci-testing.md)*
