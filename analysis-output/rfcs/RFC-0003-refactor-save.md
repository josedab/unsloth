# RFC-0003: Refactor save.py into Modular Components

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 10 person-days
**Priority:** P0 (Strategic)

---

## Summary

Refactor the monolithic `save.py` (2,954 LOC) into modular components with functions under 100 LOC and cyclomatic complexity under 15. This will improve maintainability, testability, and reduce bug rate.

---

## Motivation

The `unsloth_save_model()` function in `save.py` has a cyclomatic complexity of **118** (target: 15) and spans **598 lines**. This is the most critical maintainability issue in the codebase.

### Current Problem

```python
# From unsloth/save.py:227 - simplified structure
def unsloth_save_model(
    model, tokenizer, save_directory, save_method, ...
):
    # 598 lines handling ALL of:
    # - GGUF conversion
    # - Merged model saving
    # - Hub pushing
    # - LoRA conversion
    # - Multiple quantization methods
    # - Error handling
    # - Validation
    # ... in ONE function
```

### Impact

- **Cannot unit test:** Complexity too high for meaningful tests
- **Bug-prone:** Changes risk unintended side effects
- **Hard to understand:** 598 lines to read for any change
- **Difficult to extend:** Adding new save formats is risky
- **13 bare except clauses:** Errors hidden within this function

### Complexity Breakdown

The function handles:
1. Merged model saving (16-bit, 4-bit)
2. GGUF conversion (10+ quantization methods)
3. LoRA to GGML conversion
4. HuggingFace Hub pushing
5. Ollama model creation
6. Sharding and indexing
7. Metadata generation

Each deserves its own module.

---

## Detailed Design

### New Module Structure

```
unsloth/
├── save/
│   ├── __init__.py          # Public exports
│   ├── base.py               # SaveConfig, common utilities
│   ├── merged.py             # Merged model saving
│   ├── gguf.py               # GGUF conversion
│   ├── lora.py               # LoRA-specific saves
│   ├── hub.py                # HuggingFace Hub operations
│   ├── ollama.py             # Ollama integration
│   └── validation.py         # Input validation
```

### Base Classes and Config

```python
# unsloth/save/base.py
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path

@dataclass
class SaveConfig:
    """Configuration for model saving."""
    save_directory: Path
    push_to_hub: bool = False
    repo_id: Optional[str] = None
    token: Optional[str] = None
    private: bool = False

@dataclass
class GGUFConfig(SaveConfig):
    """Configuration for GGUF export."""
    quantization_method: str = "q4_k_m"
    first_conversion: Optional[str] = None

@dataclass
class MergedConfig(SaveConfig):
    """Configuration for merged model saving."""
    save_method: Literal["merged_16bit", "merged_4bit"] = "merged_16bit"
    maximum_memory_usage: float = 0.8

class SaveError(Exception):
    """Base exception for save operations."""
    pass

class ValidationError(SaveError):
    """Invalid configuration or input."""
    pass

class ConversionError(SaveError):
    """Model conversion failed."""
    pass
```

### Merged Model Module

```python
# unsloth/save/merged.py
from typing import TYPE_CHECKING
from .base import MergedConfig, SaveError
from .validation import validate_model, validate_directory

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

def save_merged_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: MergedConfig,
) -> Path:
    """
    Save model with LoRA weights merged into base.

    Args:
        model: Trained model (PEFT or base)
        tokenizer: Associated tokenizer
        config: Save configuration

    Returns:
        Path to saved model directory

    Raises:
        ValidationError: Invalid model or config
        SaveError: Save operation failed
    """
    # Validate inputs
    validate_model(model)
    validate_directory(config.save_directory)

    # Merge LoRA weights
    merged_model = _merge_lora_weights(model)

    # Save based on precision
    if config.save_method == "merged_16bit":
        return _save_16bit(merged_model, tokenizer, config)
    else:
        return _save_4bit(merged_model, tokenizer, config)


def _merge_lora_weights(model: "PreTrainedModel") -> "PreTrainedModel":
    """Merge LoRA adapters into base model."""
    from peft import PeftModel

    if not isinstance(model, PeftModel):
        return model

    # Merge and unload
    merged = model.merge_and_unload()

    return merged


def _save_16bit(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: MergedConfig,
) -> Path:
    """Save merged model in 16-bit precision."""
    save_path = config.save_directory / "merged_16bit"
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return save_path


def _save_4bit(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: MergedConfig,
) -> Path:
    """Save merged model with 4-bit quantization."""
    # Implementation for 4-bit saving
    ...
```

### GGUF Module

```python
# unsloth/save/gguf.py
from typing import TYPE_CHECKING, Optional
from pathlib import Path
from .base import GGUFConfig, ConversionError
from .validation import validate_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

# Quantization method registry
QUANT_METHODS = {
    "q4_0": "Q4_0",
    "q4_1": "Q4_1",
    "q4_k_m": "Q4_K_M",
    "q4_k_s": "Q4_K_S",
    "q5_0": "Q5_0",
    "q5_1": "Q5_1",
    "q5_k_m": "Q5_K_M",
    "q5_k_s": "Q5_K_S",
    "q8_0": "Q8_0",
    "f16": "F16",
}

def save_to_gguf(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: GGUFConfig,
) -> Path:
    """
    Convert and save model to GGUF format.

    Args:
        model: Model to convert
        tokenizer: Associated tokenizer
        config: GGUF save configuration

    Returns:
        Path to GGUF file

    Raises:
        ConversionError: GGUF conversion failed
        ValidationError: Invalid configuration
    """
    validate_model(model)
    _validate_gguf_config(config)

    # Step 1: Save as temporary safetensors
    temp_path = _save_temporary_model(model, tokenizer)

    try:
        # Step 2: Convert to GGUF
        gguf_path = _convert_to_gguf(temp_path, config)

        # Step 3: Quantize if needed
        if config.quantization_method != "f16":
            gguf_path = _quantize_gguf(gguf_path, config)

        return gguf_path

    finally:
        # Cleanup temporary files
        _cleanup_temp_files(temp_path)


def _validate_gguf_config(config: GGUFConfig) -> None:
    """Validate GGUF configuration."""
    if config.quantization_method not in QUANT_METHODS:
        valid = ", ".join(QUANT_METHODS.keys())
        raise ValidationError(
            f"Invalid quantization method: {config.quantization_method}. "
            f"Valid options: {valid}"
        )


def _convert_to_gguf(model_path: Path, config: GGUFConfig) -> Path:
    """Convert model to GGUF format using llama.cpp."""
    import subprocess

    output_path = config.save_directory / "model.gguf"

    try:
        subprocess.run(
            ["python", "convert.py", str(model_path), "--outfile", str(output_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise ConversionError(f"GGUF conversion failed: {e.stderr}") from e

    return output_path


def _quantize_gguf(gguf_path: Path, config: GGUFConfig) -> Path:
    """Quantize GGUF file."""
    # Implementation for quantization
    ...
```

### Hub Module

```python
# unsloth/save/hub.py
from typing import TYPE_CHECKING, Optional
from pathlib import Path
from .base import SaveConfig, SaveError

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

def push_to_hub(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: SaveConfig,
) -> str:
    """
    Push model to HuggingFace Hub.

    Args:
        model: Model to push
        tokenizer: Associated tokenizer
        config: Hub configuration

    Returns:
        Hub URL

    Raises:
        SaveError: Push failed
    """
    from huggingface_hub import HfApi

    if not config.repo_id:
        raise SaveError("repo_id required for Hub push")

    api = HfApi(token=config.token)

    # Create repo if needed
    _create_repo_if_needed(api, config)

    # Upload model
    model.push_to_hub(
        config.repo_id,
        token=config.token,
        private=config.private,
    )

    # Upload tokenizer
    tokenizer.push_to_hub(
        config.repo_id,
        token=config.token,
    )

    return f"https://huggingface.co/{config.repo_id}"
```

### Public API

```python
# unsloth/save/__init__.py
from .merged import save_merged_model
from .gguf import save_to_gguf
from .hub import push_to_hub
from .lora import save_lora_model
from .ollama import create_ollama_model
from .base import (
    SaveConfig,
    GGUFConfig,
    MergedConfig,
    SaveError,
    ValidationError,
    ConversionError,
)

__all__ = [
    "save_merged_model",
    "save_to_gguf",
    "push_to_hub",
    "save_lora_model",
    "create_ollama_model",
    "SaveConfig",
    "GGUFConfig",
    "MergedConfig",
    "SaveError",
    "ValidationError",
    "ConversionError",
]
```

---

## Implementation Plan

### Phase 1: Infrastructure (Days 1-2)
1. Create `unsloth/save/` directory structure
2. Implement base classes and configs
3. Add validation module
4. Set up tests for new structure

### Phase 2: Extract GGUF (Days 3-4)
1. Extract GGUF conversion to `gguf.py`
2. Add quantization method registry
3. Write unit tests for GGUF path
4. Validate against existing behavior

### Phase 3: Extract Merged (Days 5-6)
1. Extract merged saving to `merged.py`
2. Handle 16-bit and 4-bit paths
3. Write unit tests
4. Validate output compatibility

### Phase 4: Extract Hub/Ollama (Days 7-8)
1. Extract Hub operations to `hub.py`
2. Extract Ollama to `ollama.py`
3. Write unit tests
4. End-to-end testing

### Phase 5: Integration and Cleanup (Days 9-10)
1. Update imports throughout codebase
2. Add backward-compatible wrappers
3. Remove old `save.py`
4. Update documentation
5. Final testing

### Milestones

| Day | Deliverable | Complexity Target |
|-----|-------------|-------------------|
| 2 | Infrastructure | N/A |
| 4 | GGUF module | <15 |
| 6 | Merged module | <15 |
| 8 | Hub/Ollama | <15 |
| 10 | Integration complete | All <15 |

---

## Backwards Compatibility

### Public API Preservation

The current public functions will remain as thin wrappers:

```python
# unsloth/save.py (deprecated, for compatibility)
from .save import save_merged_model, save_to_gguf, MergedConfig, GGUFConfig

def unsloth_save_model(model, tokenizer, save_directory, save_method, ...):
    """
    Deprecated: Use save_merged_model() or save_to_gguf() directly.
    """
    import warnings
    warnings.warn(
        "unsloth_save_model is deprecated, use save_merged_model() or save_to_gguf()",
        DeprecationWarning
    )

    if save_method in ("merged_16bit", "merged_4bit"):
        config = MergedConfig(save_directory=save_directory, save_method=save_method)
        return save_merged_model(model, tokenizer, config)
    else:
        config = GGUFConfig(save_directory=save_directory, quantization_method=save_method)
        return save_to_gguf(model, tokenizer, config)
```

### Migration Path

1. **Version N:** Introduce new modules, keep old function
2. **Version N+1:** Deprecation warning on old function
3. **Version N+2:** Remove old function

---

## Alternatives Considered

### Alternative 1: Keep Single File, Extract Functions

Break into functions within `save.py`:

```python
# Still in save.py
def _save_merged(...): ...
def _save_gguf(...): ...
def _push_to_hub(...): ...
```

**Rejected:** Still too large, no separation of concerns.

### Alternative 2: Strategy Pattern

```python
class SaveStrategy:
    def save(self, model, tokenizer, path): ...

class GGUFStrategy(SaveStrategy): ...
class MergedStrategy(SaveStrategy): ...
```

**Considered:** Good pattern, but config classes are simpler for this use case.

### Alternative 3: Builder Pattern

```python
SaveBuilder(model, tokenizer)
    .to_gguf()
    .quantize("q4_k_m")
    .push_to_hub("repo")
    .save()
```

**Considered for future:** Nice API, but larger change.

---

## Open Questions

1. **Should we support plugins for new save formats?** Could add a registry.

2. **How to handle combined operations?** E.g., merge + GGUF + push. Pipeline pattern?

3. **Where should temporary files go?** Currently uses `tempfile`, should be configurable.

4. **Should we add progress callbacks?** For long-running operations.

---

## Success Criteria

- [ ] All functions have complexity < 15
- [ ] No function exceeds 100 LOC
- [ ] 80%+ test coverage for save modules
- [ ] All existing tests pass
- [ ] Output files identical to old implementation
- [ ] 0 bare except clauses

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Max complexity | 118 | <15 |
| Max function LOC | 598 | <100 |
| Test coverage | ~0% | 80%+ |
| Bare excepts | 13 | 0 |

---

## Required Approvals

- [ ] Maintainer review for architecture
- [ ] Full test suite pass
- [ ] Output compatibility verified
- [ ] Documentation updated

---

## Rollback Strategy

1. Keep old `save.py` in tree during transition
2. Feature flag to use old vs new implementation
3. Can revert to old implementation if issues found
4. No data format changes, so no migration needed

---

## References

- Martin Fowler, "Refactoring" - Extract Class pattern
- "Code Complete" by Steve McConnell - Function length guidelines
- Unsloth save.py: https://github.com/unslothai/unsloth/blob/341ce85864d191e4a6b7c447b9167c1faf5e20d3/unsloth/save.py

---

*Next: [RFC-0004: Implement Structured Logging](./RFC-0004-structured-logging.md)*
