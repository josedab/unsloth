# RFC-0002: Add Type Hints to Public API

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 3 person-days
**Priority:** P0 (Quick Win)

---

## Summary

Add comprehensive type hints to all public API functions, achieving 80%+ coverage for the public interface. This enables IDE support, static analysis, and better documentation.

---

## Motivation

Currently only 13% of functions (81 of 625) have return type hints. The public API has almost no type information, making it difficult for users to understand expected inputs and outputs.

### Current Problem

```python
# From unsloth/models/loader.py - no types
def from_pretrained(model_name, max_seq_length=None, dtype=None, load_in_4bit=True):
    ...
```

Users must guess:
- What type is `model_name`? (str? Path?)
- What's returned? (tuple? model?)
- What are valid `dtype` values?

### Impact

- **No IDE autocomplete:** Reduces developer productivity
- **No static analysis:** mypy/pyright can't catch errors
- **Poor documentation:** Types are not self-documenting
- **Harder onboarding:** New contributors must read implementation

### Worst Offenders

| Module | Functions | With Types | Coverage |
|--------|-----------|------------|----------|
| `vision.py` | 47 | 0 | 0% |
| `tokenizer_utils.py` | 32 | 0 | 0% |
| `llama.py` | 49 | 5 | 10.2% |

---

## Detailed Design

### Scope: Public API Only

Focus on user-facing functions:
- `FastLanguageModel.from_pretrained()`
- `FastLanguageModel.get_peft_model()`
- `UnslothTrainer`
- `save_to_gguf()`
- All public functions in `__init__.py` exports

### Type Definitions

Create a `types.py` module for shared types:

```python
# unsloth/types.py
from typing import (
    Optional, Union, Tuple, List, Dict, Any,
    Literal, TypeVar, TYPE_CHECKING
)
from pathlib import Path

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from peft import PeftModel

# Type aliases
ModelName = Union[str, Path]
DType = Optional[Literal["float16", "bfloat16", "float32"]]
DeviceMap = Union[str, Dict[str, Union[int, str]]]
QuantizationMethod = Literal[
    "q4_0", "q4_1", "q4_k_m", "q4_k_s",
    "q5_0", "q5_1", "q5_k_m", "q5_k_s",
    "q8_0", "f16"
]

# Return types
ModelTokenizerTuple = Tuple["PreTrainedModel", "PreTrainedTokenizer"]
```

### FastLanguageModel Types

```python
# unsloth/models/loader.py
from ..types import ModelName, DType, DeviceMap, ModelTokenizerTuple

class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: ModelName,
        max_seq_length: Optional[int] = None,
        dtype: DType = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: DeviceMap = "auto",
        trust_remote_code: bool = False,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelTokenizerTuple:
        """
        Load a model with Unsloth optimizations.

        Args:
            model_name: HuggingFace model ID or local path
            max_seq_length: Maximum sequence length (default: from config)
            dtype: Data type for computation
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            device_map: Device placement strategy
            trust_remote_code: Allow custom model code
            token: HuggingFace API token

        Returns:
            Tuple of (model, tokenizer)

        Example:
            >>> model, tokenizer = FastLanguageModel.from_pretrained(
            ...     "unsloth/llama-3-8b-bnb-4bit",
            ...     max_seq_length=2048,
            ... )
        """
        ...

    @staticmethod
    def get_peft_model(
        model: "PreTrainedModel",
        r: int = 16,
        lora_alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.0,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_gradient_checkpointing: Union[bool, Literal["unsloth"]] = "unsloth",
        **kwargs: Any,
    ) -> "PeftModel":
        """
        Apply LoRA to a model.

        Args:
            model: Base model to apply LoRA to
            r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            target_modules: Modules to apply LoRA (default: attention)
            lora_dropout: Dropout probability
            bias: Bias training strategy
            use_gradient_checkpointing: Gradient checkpointing mode

        Returns:
            Model with LoRA applied
        """
        ...
```

### Trainer Types

```python
# unsloth/trainer.py
from typing import Optional, Union, Dict, Any, List
from transformers import TrainingArguments
from datasets import Dataset

class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate: Optional[float] = None

class UnslothTrainer:
    def __init__(
        self,
        model: "PreTrainedModel",
        args: Optional[UnslothTrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        data_collator: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        ...

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
    ) -> "TrainOutput":
        ...
```

### Save Function Types

```python
# unsloth/save.py
from ..types import QuantizationMethod

def save_to_gguf(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    save_directory: Union[str, Path],
    quantization_method: QuantizationMethod = "q4_k_m",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Export model to GGUF format for llama.cpp.

    Args:
        model: Trained model to export
        tokenizer: Associated tokenizer
        save_directory: Output directory
        quantization_method: GGUF quantization type
        push_to_hub: Upload to HuggingFace Hub
        repo_id: Hub repository ID
        token: Hub API token

    Returns:
        Path to saved GGUF file
    """
    ...
```

---

## Example Usage

### IDE Autocomplete

```python
from unsloth import FastLanguageModel

# IDE now shows:
# from_pretrained(
#     model_name: str | Path,
#     max_seq_length: int | None = None,
#     ...
# ) -> tuple[PreTrainedModel, PreTrainedTokenizer]

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",  # IDE knows this should be str/Path
    max_seq_length=2048,              # IDE knows this should be int
)
```

### Static Analysis

```python
# mypy will catch this error:
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama",
    max_seq_length="2048",  # Error: expected int, got str
)
```

---

## Implementation Plan

### Phase 1: Type Infrastructure (Day 1 morning)
1. Create `unsloth/types.py` with shared types
2. Set up TYPE_CHECKING imports to avoid circular imports
3. Add py.typed marker file

### Phase 2: Core API (Day 1 afternoon)
1. Type `FastLanguageModel.from_pretrained()`
2. Type `FastLanguageModel.get_peft_model()`
3. Type `UnslothTrainer` and `UnslothTrainingArguments`

### Phase 3: Save Functions (Day 2 morning)
1. Type `save_to_gguf()`
2. Type `save_pretrained_merged()`
3. Type `push_to_hub_merged()`

### Phase 4: Secondary API (Day 2 afternoon)
1. Type `chat_templates.py` functions
2. Type `tokenizer_utils.py` functions
3. Type registry functions

### Phase 5: Validation (Day 3)
1. Run mypy on typed modules
2. Test IDE autocomplete
3. Update documentation with types
4. Add type-checking to pre-commit (optional)

### Milestones

| Day | Deliverable | Coverage |
|-----|-------------|----------|
| 1 | Core API typed | 50% |
| 2 | Save/utils typed | 75% |
| 3 | Validation complete | 80%+ |

---

## Backwards Compatibility

### Breaking Changes
None. Type hints are purely additive in Python.

### Runtime Behavior
Types are only used by static analyzers; no runtime impact.

### Python Version Support
Type syntax compatible with Python 3.9+ (current minimum).

---

## Alternatives Considered

### Alternative 1: Stub Files (.pyi)

Create separate `.pyi` files with types:

```python
# unsloth/models/loader.pyi
class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        ...
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: ...
```

**Rejected:** Harder to maintain, types can drift from implementation.

### Alternative 2: Full Codebase Typing

Type all 625 functions.

**Rejected for now:** Too large scope for initial RFC. Start with public API, expand later.

### Alternative 3: Use typing_extensions

For advanced features like `ParamSpec`:

```python
from typing_extensions import ParamSpec
P = ParamSpec('P')
```

**Considered:** May use for specific features, not core scope.

---

## Open Questions

1. **Should we enforce types in CI?** Could add mypy to pre-commit.

2. **What about runtime type checking?** Libraries like `beartype` could validate at runtime.

3. **How to type model-specific parameters?** E.g., Llama vs Mistral configs.

4. **Should we use `@overload` for different return types?**

---

## Success Criteria

- [ ] 80%+ type coverage for public API functions
- [ ] py.typed marker added for PEP 561
- [ ] mypy passes on typed modules
- [ ] IDE autocomplete works correctly
- [ ] No runtime behavior changes

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Public API type coverage | 13% | 80%+ |
| IDE autocomplete | No | Yes |
| Static analysis errors caught | 0 | Many |

---

## Required Approvals

- [ ] Maintainer review
- [ ] mypy clean
- [ ] IDE testing verification

---

## Rollback Strategy

Type hints can be removed without breaking runtime behavior. Individual modules can be reverted if issues arise.

---

## References

- PEP 484: Type Hints - https://peps.python.org/pep-0484/
- PEP 561: Distributing and Packaging Type Information - https://peps.python.org/pep-0561/
- mypy documentation - https://mypy.readthedocs.io/
- HuggingFace transformers types - Example of typed ML library

---

*Next: [RFC-0003: Refactor save.py into Modular Components](./RFC-0003-refactor-save.md)*
