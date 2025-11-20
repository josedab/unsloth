# RFC-0013: Enhanced Error Messages with Troubleshooting

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 7 person-days
**Priority:** P1 (Strategic)

---

## Summary

Transform cryptic error messages into actionable diagnostics with context, troubleshooting steps, and direct links to solutions, reducing support burden and improving user experience.

---

## Motivation

Users currently encounter unhelpful error messages that require deep knowledge to debug:

### Current Error Messages

**Example 1: OOM Error**
```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

User doesn't know:
- Which operation caused this
- How to reduce memory
- Optimal batch size for their GPU

**Example 2: Incompatible Model**
```python
KeyError: 'model_type'
```

User doesn't know:
- What model type was expected
- Which models are supported
- How to check compatibility

**Example 3: Quantization Error**
```python
AssertionError
```

User doesn't know:
- What assertion failed
- Why it failed
- How to fix it

### Impact

| Issue | Effect |
|-------|--------|
| Cryptic errors | Users get stuck |
| No troubleshooting | High support burden |
| No context | Can't debug themselves |
| No documentation links | Can't find solutions |

GitHub issues show repeated questions about the same errors.

---

## Detailed Design

### Error Message Framework

```python
# unsloth/errors.py
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class TroubleshootingStep:
    """A single troubleshooting step."""
    action: str
    command: Optional[str] = None
    explanation: str = ""

class UnslothError(Exception):
    """
    Base exception with enhanced error messaging.

    Provides:
    - Clear error description
    - Context about what went wrong
    - Actionable troubleshooting steps
    - Links to documentation

    Example:
        raise UnslothError(
            message="Model not found",
            context={"model_name": "invalid/model"},
            troubleshooting=[
                TroubleshootingStep(
                    action="Check model name spelling",
                    explanation="Verify the model ID on HuggingFace"
                ),
            ],
            docs_url="https://docs.unsloth.ai/models"
        )
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict] = None,
        troubleshooting: Optional[List[TroubleshootingStep]] = None,
        docs_url: Optional[str] = None,
        support_url: Optional[str] = None,
    ):
        self.message = message
        self.context = context or {}
        self.troubleshooting = troubleshooting or []
        self.docs_url = docs_url
        self.support_url = support_url

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format comprehensive error message."""
        lines = [
            "",
            "=" * 70,
            f"❌ {self.message}",
            "=" * 70,
        ]

        # Add context
        if self.context:
            lines.append("")
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")

        # Add troubleshooting steps
        if self.troubleshooting:
            lines.append("")
            lines.append("Troubleshooting Steps:")
            for i, step in enumerate(self.troubleshooting, 1):
                lines.append(f"  {i}. {step.action}")
                if step.command:
                    lines.append(f"     $ {step.command}")
                if step.explanation:
                    lines.append(f"     → {step.explanation}")

        # Add documentation link
        if self.docs_url:
            lines.append("")
            lines.append(f"📖 Documentation: {self.docs_url}")

        # Add support link
        if self.support_url:
            lines.append(f"💬 Get help: {self.support_url}")

        lines.append("=" * 70)

        return "\n".join(lines)

class ModelNotFoundError(UnslothError):
    """Model not found on HuggingFace or locally."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            context={
                "model_name": model_name,
                "search_locations": [
                    "HuggingFace Hub",
                    "Local cache",
                ],
            },
            troubleshooting=[
                TroubleshootingStep(
                    action="Verify model name on HuggingFace",
                    command=f"huggingface-cli repo-info {model_name}",
                    explanation="Check if the model exists and you have access",
                ),
                TroubleshootingStep(
                    action="Check for typos in model name",
                    explanation="Model names are case-sensitive",
                ),
                TroubleshootingStep(
                    action="Verify internet connection",
                    command="ping huggingface.co",
                    explanation="Required for downloading models",
                ),
                TroubleshootingStep(
                    action="Check authentication if model is private",
                    command="huggingface-cli login",
                ),
            ],
            docs_url="https://docs.unsloth.ai/models/loading",
            support_url="https://github.com/unslothai/unsloth/issues",
        )

class OutOfMemoryError(UnslothError):
    """CUDA out of memory error with diagnostics."""

    def __init__(
        self,
        requested_mb: float,
        available_mb: float,
        allocated_mb: float,
        operation: str = "unknown",
    ):
        # Calculate recommendations
        reduction_needed = (requested_mb - available_mb) / allocated_mb * 100

        troubleshooting = [
            TroubleshootingStep(
                action="Reduce batch size",
                explanation=f"Try reducing by {int(reduction_needed)}%",
            ),
        ]

        if reduction_needed > 50:
            troubleshooting.append(
                TroubleshootingStep(
                    action="Enable gradient checkpointing",
                    command="model.gradient_checkpointing_enable()",
                    explanation="Trades compute for memory (30-50% reduction)",
                )
            )

        troubleshooting.extend([
            TroubleshootingStep(
                action="Reduce sequence length",
                explanation="Memory usage scales quadratically with length",
            ),
            TroubleshootingStep(
                action="Use automatic batch size finder",
                command="unsloth find-batch-size",
                explanation="Automatically finds optimal batch size",
            ),
        ])

        super().__init__(
            message=f"Out of memory during {operation}",
            context={
                "requested_mb": f"{requested_mb:.1f} MB",
                "available_mb": f"{available_mb:.1f} MB",
                "allocated_mb": f"{allocated_mb:.1f} MB",
                "reduction_needed": f"{reduction_needed:.0f}%",
            },
            troubleshooting=troubleshooting,
            docs_url="https://docs.unsloth.ai/troubleshooting/oom",
        )

class UnsupportedModelError(UnslothError):
    """Model architecture not supported."""

    def __init__(self, model_type: str, supported_types: List[str]):
        super().__init__(
            message=f"Model type '{model_type}' is not supported",
            context={
                "model_type": model_type,
                "supported_types": ", ".join(supported_types),
            },
            troubleshooting=[
                TroubleshootingStep(
                    action="Use a supported model",
                    explanation=f"Supported: {', '.join(supported_types[:5])}",
                ),
                TroubleshootingStep(
                    action="Check if model is compatible",
                    command="unsloth check-model <model-name>",
                ),
                TroubleshootingStep(
                    action="Request support for this model",
                    explanation="Open an issue on GitHub",
                ),
            ],
            docs_url="https://docs.unsloth.ai/models/supported",
            support_url="https://github.com/unslothai/unsloth/issues/new?template=model_request.md",
        )

class QuantizationError(UnslothError):
    """Quantization failed."""

    def __init__(self, reason: str, model_name: str):
        super().__init__(
            message=f"Quantization failed: {reason}",
            context={
                "model_name": model_name,
                "reason": reason,
            },
            troubleshooting=[
                TroubleshootingStep(
                    action="Check bitsandbytes installation",
                    command="pip install bitsandbytes>=0.41.0",
                ),
                TroubleshootingStep(
                    action="Verify GPU compute capability",
                    explanation="4-bit quantization requires compute capability 7.0+",
                ),
                TroubleshootingStep(
                    action="Try 8-bit quantization instead",
                    command='load_in_8bit=True, load_in_4bit=False',
                ),
            ],
            docs_url="https://docs.unsloth.ai/quantization",
        )
```

### Error Context Decorator

```python
# unsloth/errors.py (continued)

from functools import wraps
import torch

def with_error_context(operation_name: str):
    """
    Decorator to add context to errors.

    Example:
        @with_error_context("model loading")
        def load_model(name):
            # Any errors get wrapped with context
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Convert to enhanced OOM error
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    available = max_memory - allocated

                    raise OutOfMemoryError(
                        requested_mb=reserved - allocated,
                        available_mb=available,
                        allocated_mb=allocated,
                        operation=operation_name,
                    ) from e

                # Re-raise with context
                raise UnslothError(
                    message=f"Error during {operation_name}",
                    context={
                        "operation": operation_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                ) from e

            except KeyError as e:
                # Common in model loading
                raise UnslothError(
                    message=f"Missing key during {operation_name}",
                    context={
                        "operation": operation_name,
                        "missing_key": str(e),
                    },
                    troubleshooting=[
                        TroubleshootingStep(
                            action="Verify model format",
                            explanation="Model may be corrupted or incompatible",
                        ),
                        TroubleshootingStep(
                            action="Re-download model",
                            command="rm -rf ~/.cache/huggingface/hub/<model>",
                        ),
                    ],
                ) from e

        return wrapper
    return decorator
```

### Integration Example

```python
# unsloth/models/loader.py (updated)
from unsloth.errors import (
    with_error_context,
    ModelNotFoundError,
    UnsupportedModelError,
)

@with_error_context("model loading")
def from_pretrained(model_name: str, **kwargs):
    """Load model with enhanced error handling."""

    # Check if model exists
    if not model_exists(model_name):
        raise ModelNotFoundError(model_name)

    # Check if supported
    model_type = get_model_type(model_name)
    if model_type not in SUPPORTED_MODELS:
        raise UnsupportedModelError(
            model_type=model_type,
            supported_types=list(SUPPORTED_MODELS.keys()),
        )

    # Load model (any errors get wrapped with context)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    return model
```

### CLI Error Checker

```python
# unsloth/cli/check.py
import click
from unsloth import FastLanguageModel
from unsloth.errors import UnslothError

@click.command()
@click.argument("model_name")
def check_model(model_name):
    """
    Check if a model is compatible with Unsloth.

    Example:
        unsloth check-model meta-llama/Llama-3-8B
    """
    click.echo(f"Checking model: {model_name}")
    click.echo("=" * 50)

    try:
        # Try to load model info
        from unsloth.models._utils import get_model_info

        info = get_model_info(model_name)

        click.echo("✓ Model found")
        click.echo(f"  Type: {info.model_type}")
        click.echo(f"  Quantization: {info.quantization}")
        click.echo(f"  Supported: Yes")

    except UnslothError as e:
        # Error is already formatted
        click.echo(str(e))
        return 1

    except Exception as e:
        click.echo(f"✗ Unexpected error: {e}")
        return 1

    return 0
```

---

## Example Usage

### Before (Cryptic Error)

```python
>>> model, tokenizer = FastLanguageModel.from_pretrained("invalid/model")
KeyError: 'config.json'
```

### After (Enhanced Error)

```python
>>> model, tokenizer = FastLanguageModel.from_pretrained("invalid/model")

======================================================================
❌ Model 'invalid/model' not found
======================================================================

Context:
  model_name: invalid/model
  search_locations: ['HuggingFace Hub', 'Local cache']

Troubleshooting Steps:
  1. Verify model name on HuggingFace
     $ huggingface-cli repo-info invalid/model
     → Check if the model exists and you have access

  2. Check for typos in model name
     → Model names are case-sensitive

  3. Verify internet connection
     $ ping huggingface.co
     → Required for downloading models

  4. Check authentication if model is private
     $ huggingface-cli login

📖 Documentation: https://docs.unsloth.ai/models/loading
💬 Get help: https://github.com/unslothai/unsloth/issues
======================================================================
```

### OOM Error Example

```python
>>> trainer.train()

======================================================================
❌ Out of memory during forward pass
======================================================================

Context:
  requested_mb: 2048.0 MB
  available_mb: 1024.0 MB
  allocated_mb: 14336.0 MB
  reduction_needed: 50%

Troubleshooting Steps:
  1. Reduce batch size
     → Try reducing by 50%

  2. Enable gradient checkpointing
     $ model.gradient_checkpointing_enable()
     → Trades compute for memory (30-50% reduction)

  3. Reduce sequence length
     → Memory usage scales quadratically with length

  4. Use automatic batch size finder
     $ unsloth find-batch-size
     → Automatically finds optimal batch size

📖 Documentation: https://docs.unsloth.ai/troubleshooting/oom
======================================================================
```

---

## Implementation Plan

### Phase 1: Error Framework (Days 1-2)

**Day 1: Base Classes**
- Implement UnslothError
- Implement TroubleshootingStep
- Add formatting logic

**Day 2: Common Errors**
- ModelNotFoundError
- OutOfMemoryError
- UnsupportedModelError
- QuantizationError

### Phase 2: Integration (Days 3-5)

**Day 3: Model Loading**
- Add error context to from_pretrained()
- Add error context to get_peft_model()
- Test error messages

**Day 4: Training**
- Add error context to trainer
- Add error context to save functions
- Test error messages

**Day 5: Edge Cases**
- Handle all common errors
- Add fallback for unknown errors
- Test comprehensive coverage

### Phase 3: Documentation and Tools (Days 6-7)

**Day 6: CLI Tools**
- Implement check-model command
- Implement diagnose command
- Test CLI tools

**Day 7: Documentation**
- Write troubleshooting guide
- Document all error types
- Create error reference

### Milestones

| Day | Deliverable |
|-----|-------------|
| 2 | Error framework complete |
| 5 | Integration complete |
| 7 | Documentation and tools |

---

## Backwards Compatibility

### Breaking Changes

None. Enhanced errors are compatible with existing error handling.

### Error Handling

```python
# Existing code still works
try:
    model = FastLanguageModel.from_pretrained(...)
except Exception as e:
    print(f"Error: {e}")  # Shows enhanced message
```

---

## Alternatives Considered

### Alternative 1: Use Existing Exceptions

Keep current error messages.

**Rejected:**
- Poor user experience
- High support burden
- Users get stuck

### Alternative 2: External Error Database

Store error messages in separate database.

**Rejected:**
- Adds complexity
- Errors may become outdated
- Requires network access

### Alternative 3: Error Codes

Use error codes instead of messages.

**Rejected:**
- Less user-friendly
- Requires looking up codes
- Extra indirection

---

## Open Questions

1. **Should errors include telemetry?**
   - Pro: Better debugging
   - Con: Privacy concerns

2. **Support error translations?**
   - Large user base speaks many languages
   - Significant maintenance

3. **Include stack traces by default?**
   - Pro: More debugging info
   - Con: Can be overwhelming

---

## Success Criteria

- [ ] All common errors have enhanced messages
- [ ] Troubleshooting steps are actionable
- [ ] Documentation links work
- [ ] Support requests reduced by 30%+
- [ ] User satisfaction improved

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Errors with troubleshooting | 0% | 80%+ |
| Support requests | High | 30% reduction |
| User satisfaction | Low | High |
| Time to resolution | Hours | Minutes |

---

## Required Approvals

- [ ] User experience review
- [ ] Documentation review
- [ ] Support team feedback

---

## Rollback Strategy

1. Enhanced errors are backward compatible
2. Can disable with environment variable
3. Falls back to original errors

---

## References

- Python Exception Hierarchy: https://docs.python.org/3/library/exceptions.html
- Error Message Best Practices: https://www.nngroup.com/articles/error-message-guidelines/
- Rust Error Handling: https://doc.rust-lang.org/book/ch09-00-error-handling.html

---

*Next: [RFC-0014: Intelligent Model Caching and Disk Management](./RFC-0014-model-caching.md)*
