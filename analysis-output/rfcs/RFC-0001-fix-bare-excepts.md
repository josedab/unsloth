# RFC-0001: Fix Bare Except Clauses

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 2 person-days
**Priority:** P0 (Quick Win)

---

## Summary

Replace all 39 bare `except:` clauses with specific exception handling to improve debuggability and error visibility across the Unsloth codebase.

---

## Motivation

The codebase contains 39 bare `except:` clauses that silently swallow all exceptions, including critical errors like `KeyboardInterrupt`, `SystemExit`, and `MemoryError`. This makes debugging extremely difficult as errors are hidden.

### Current Problem

```python
# From unsloth/models/_utils.py - actual code
try:
    apply_optimization()
except:
    pass  # What went wrong? We'll never know.
```

### Impact

- **Debugging time increased:** Errors silently ignored
- **Hidden failures:** Optimizations may not apply
- **Poor user experience:** Users don't know why things aren't working
- **Maintenance burden:** Contributors can't diagnose issues

### Affected Files

| File | Count | Severity |
|------|-------|----------|
| `unsloth/models/_utils.py` | 39 | Critical |
| `unsloth/save.py` | 13 | High |
| `unsloth/models/vision.py` | 13 | High |
| `unsloth/tokenizer_utils.py` | 8 | Medium |

---

## Detailed Design

### Principle: Graceful Degradation with Visibility

Replace bare excepts with:
1. Specific exception types
2. Logging of the failure
3. Documented fallback behavior

### Pattern 1: Optional Optimization

For optimizations that can safely be skipped:

```python
# Before
try:
    apply_flash_attention()
except:
    pass

# After
try:
    apply_flash_attention()
except ImportError as e:
    logger.debug(f"Flash attention not available: {e}, using standard attention")
except RuntimeError as e:
    logger.warning(f"Flash attention failed: {e}, falling back to standard")
```

### Pattern 2: Required Operation

For operations that must succeed:

```python
# Before
try:
    load_model_weights()
except:
    pass  # Model will be broken!

# After
try:
    load_model_weights()
except (FileNotFoundError, PermissionError) as e:
    raise RuntimeError(f"Unsloth: Failed to load model weights: {e}") from e
except Exception as e:
    logger.error(f"Unexpected error loading weights: {e}")
    raise
```

### Pattern 3: Import Fallback

For optional imports:

```python
# Before
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except:
    HAS_FLASH_ATTN = False

# After
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    logger.debug("flash_attn not installed, will use standard attention")
```

### Implementation for _utils.py

The 39 bare excepts in `_utils.py` fall into categories:

#### Category A: Import checks (15 instances)
```python
# Replace with specific ImportError
except ImportError:
    pass
```

#### Category B: Optional patches (12 instances)
```python
# Replace with logged warning
except Exception as e:
    logger.debug(f"Optional patch {patch_name} not applied: {e}")
```

#### Category C: Compatibility checks (8 instances)
```python
# Replace with specific error types
except (AttributeError, TypeError) as e:
    # Expected for older versions
    pass
```

#### Category D: Critical operations (4 instances)
```python
# Replace with re-raise after logging
except Exception as e:
    logger.error(f"Critical operation failed: {e}")
    raise
```

---

## Example Usage

### Before

```python
# unsloth/models/_utils.py:1234
def patch_model_for_training(model):
    try:
        patch_attention(model)
    except:
        pass

    try:
        patch_mlp(model)
    except:
        pass

    return model
```

### After

```python
# unsloth/models/_utils.py:1234
def patch_model_for_training(model):
    try:
        patch_attention(model)
    except ImportError as e:
        logger.info(f"Unsloth: Attention optimization unavailable: {e}")
    except Exception as e:
        logger.warning(f"Unsloth: Attention patch failed: {e}, using default")

    try:
        patch_mlp(model)
    except ImportError as e:
        logger.info(f"Unsloth: MLP optimization unavailable: {e}")
    except Exception as e:
        logger.warning(f"Unsloth: MLP patch failed: {e}, using default")

    return model
```

---

## Implementation Plan

### Phase 1: Audit and Categorize (Day 1 morning)
1. List all 39 bare excepts with context
2. Categorize each by pattern (A/B/C/D)
3. Document expected exceptions for each

### Phase 2: Implement Fixes (Day 1 afternoon - Day 2 morning)
1. Start with `_utils.py` (most critical)
2. Add logging infrastructure if needed
3. Replace each bare except with specific handling
4. Add tests for exception paths

### Phase 3: Test and Validate (Day 2 afternoon)
1. Run existing test suite
2. Test failure scenarios manually
3. Verify logging output
4. Update documentation

### Milestones

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| 1 AM | Audit complete | Categorized list |
| 1 PM | _utils.py fixed | PR ready for review |
| 2 AM | Other files fixed | All PRs ready |
| 2 PM | Testing complete | Merged to main |

---

## Backwards Compatibility

### Breaking Changes
None expected. This improves error visibility without changing behavior.

### Potential Issues
Some users may have been relying on silent failures. If an optimization was silently failing before, it will now log a warning.

### Migration
No migration needed. Users will see more informative logs.

---

## Alternatives Considered

### Alternative 1: Keep Bare Excepts with Comments

```python
try:
    optimize()
except:  # Intentional: optimization is optional
    pass
```

**Rejected:** Still hides errors, comments don't help debugging.

### Alternative 2: Add Global Exception Handler

```python
@handle_exceptions
def optimize():
    ...
```

**Rejected:** Too invasive, harder to understand flow.

### Alternative 3: Use warnings.warn Instead of Logging

```python
except Exception as e:
    warnings.warn(f"Optimization failed: {e}")
```

**Considered:** Good for user-facing issues, but logging better for debugging.

---

## Open Questions

1. **Should we add a verbose mode?** Currently `UNSLOTH_ENABLE_LOGGING` exists but isn't widely used.

2. **What log level for optional optimization failures?** Proposed: DEBUG for expected, WARNING for unexpected.

3. **Should we create custom exception classes?** E.g., `UnslothOptimizationError`. Could be future RFC.

---

## Success Criteria

- [ ] 0 bare `except:` clauses in codebase
- [ ] All exception handlers catch specific types
- [ ] Failures are logged with actionable messages
- [ ] Existing tests pass
- [ ] No regression in performance

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Bare excepts | 39 | 0 |
| Debug time for issues | Hours | Minutes |
| User confusion from silent failures | High | Low |

---

## Required Approvals

- [ ] Maintainer review
- [ ] Test suite pass
- [ ] No performance regression

---

## Rollback Strategy

If issues arise:
1. Revert the PR
2. Individual except clauses can be reverted independently
3. No data migration needed

---

## References

- Python Exception Handling Best Practices: https://docs.python.org/3/tutorial/errors.html
- "Bare except is almost never a good idea" - Python Style Guide
- Unsloth Issue Template: Users report silent failures frequently

---

*Next: [RFC-0002: Add Type Hints to Public API](./RFC-0002-type-hints.md)*
