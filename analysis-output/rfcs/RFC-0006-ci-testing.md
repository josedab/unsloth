# RFC-0006: Automated CI Testing Pipeline

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 12 person-days
**Priority:** P1 (Strategic)

---

## Summary

Implement automated testing in CI/CD pipeline using GitHub Actions with GPU runners, achieving automated quality assurance for all PRs and preventing regressions from reaching the main branch.

---

## Motivation

Currently, Unsloth has no automated testing in CI. Only pre-commit linting runs automatically. Tests exist but are run manually by developers, leading to:

- Regressions reaching main branch undetected
- Inconsistent quality across PRs
- No coverage reporting
- Contributor uncertainty about test expectations
- Maintainer burden to manually test PRs

### Current CI State

```yaml
# Only two workflows exist:
.github/workflows/
├── pre-commit.yml    # Linting only
└── stale.yml         # Issue management
```

### Impact

| Issue | Effect |
|-------|--------|
| No automated tests | Regressions reach main |
| No coverage reports | Unknown test coverage |
| Manual testing | Slow PR review |
| No quality gates | Inconsistent quality |

---

## Detailed Design

### Test Categories

#### 1. Unit Tests (No GPU)

Fast tests that mock GPU operations:

```python
# tests/unit/test_registry.py
def test_model_lookup():
    from unsloth.registry import get_model_info

    info = get_model_info("unsloth/llama-3-8b-bnb-4bit")
    assert info.model_type == "llama"
    assert info.quantization == QuantType.BNB_4BIT

def test_invalid_model():
    from unsloth.registry import get_model_info

    with pytest.raises(ValueError):
        get_model_info("invalid/model-name")
```

#### 2. Integration Tests (GPU Required)

Tests that require actual GPU:

```python
# tests/integration/test_model_loading.py
@pytest.mark.gpu
def test_load_4bit_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=512,
    )

    assert model is not None
    assert tokenizer is not None
    assert model.config.quantization_config is not None
```

#### 3. E2E Tests (Full Workflow)

Complete training workflows:

```python
# tests/e2e/test_qlora_workflow.py
@pytest.mark.gpu
@pytest.mark.slow
def test_qlora_training():
    # Load
    model, tokenizer = FastLanguageModel.from_pretrained(...)

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(model, r=8)

    # Train
    trainer = UnslothTrainer(model=model, ...)
    trainer.train()

    # Verify
    assert trainer.state.global_step > 0
```

### CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Fast tests - run on all PRs
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest pytest-cov

      - name: Run unit tests
        run: |
          pytest tests/unit -v \
            --cov=unsloth \
            --cov-report=xml \
            --cov-report=term-missing

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  # GPU tests - run on main and labeled PRs
  integration-tests:
    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'run-gpu-tests')
    runs-on: [self-hosted, gpu]

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install -e ".[dev,triton]"

      - name: Run integration tests
        run: |
          pytest tests/integration -v \
            -m gpu \
            --timeout=300

      - name: Run E2E tests
        run: |
          pytest tests/e2e -v \
            -m "gpu and not slow" \
            --timeout=600

  # Type checking
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install mypy
      - run: mypy unsloth --ignore-missing-imports
```

### Test Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # No GPU required
│   ├── test_registry.py
│   ├── test_mapper.py
│   ├── test_config.py
│   ├── test_validation.py
│   └── test_chat_templates.py
├── integration/             # GPU required
│   ├── test_model_loading.py
│   ├── test_lora_application.py
│   ├── test_training.py
│   └── test_saving.py
└── e2e/                     # Full workflows
    ├── test_qlora_workflow.py
    ├── test_dpo_workflow.py
    └── test_gguf_export.py
```

### Shared Fixtures

```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture
def small_model():
    """Load a small model for testing."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/tinyllama-bnb-4bit",  # Small model for tests
        max_seq_length=256,
    )
    yield model, tokenizer

    # Cleanup
    del model
    torch.cuda.empty_cache()

@pytest.fixture
def sample_dataset():
    """Create a small dataset for testing."""
    from datasets import Dataset

    return Dataset.from_dict({
        "text": [
            "Hello, how are you?",
            "I am fine, thank you.",
        ] * 10
    })

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow")
```

### GPU Runner Options

| Option | Cost | Pros | Cons |
|--------|------|------|------|
| **Self-hosted** | Hardware cost | Full control, no limits | Maintenance burden |
| **GitHub GPU** | Per-minute | Integrated, managed | Limited availability |
| **Cloud (Lambda Labs)** | ~$0.50/hr | Cheap, available | Setup complexity |
| **Cloud (AWS/GCP)** | ~$1-3/hr | Reliable | Higher cost |

**Recommendation:** Start with self-hosted runner for cost control, evaluate GitHub GPU runners when available.

### Coverage Thresholds

```toml
# pyproject.toml
[tool.coverage.run]
source = ["unsloth"]
omit = ["tests/*", "unsloth/kernels/moe/*"]

[tool.coverage.report]
fail_under = 60
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## Example Usage

### Running Tests Locally

```bash
# Unit tests only (no GPU)
pytest tests/unit -v

# Integration tests (requires GPU)
pytest tests/integration -v -m gpu

# All tests with coverage
pytest tests/ -v --cov=unsloth

# Specific test
pytest tests/unit/test_registry.py::test_model_lookup -v
```

### PR Workflow

1. Developer pushes PR
2. Unit tests run automatically (~2 min)
3. If "run-gpu-tests" label added, GPU tests run (~10 min)
4. Coverage report posted to PR
5. All checks must pass to merge

---

## Implementation Plan

### Phase 1: Test Infrastructure (Days 1-3)

**Day 1: Setup**
- Create test directory structure
- Add pytest configuration
- Create conftest.py with fixtures

**Day 2: CI Configuration**
- Create GitHub Actions workflow
- Configure coverage reporting
- Set up Codecov integration

**Day 3: Unit Test Framework**
- Write first unit tests for registry
- Verify CI runs correctly
- Fix any configuration issues

### Phase 2: Unit Tests (Days 4-6)

**Day 4: Registry and Config**
- test_registry.py
- test_mapper.py
- test_config.py

**Day 5: Utilities**
- test_validation.py
- test_chat_templates.py
- test_tokenizer_utils.py

**Day 6: Mocks for GPU Code**
- Create GPU mocks
- Test model loading with mocks
- Test save functions with mocks

### Phase 3: GPU Runner Setup (Days 7-8)

**Day 7: Runner Configuration**
- Set up self-hosted runner
- Install dependencies
- Configure GPU access

**Day 8: Integration Tests**
- test_model_loading.py
- test_lora_application.py
- Verify GPU tests run in CI

### Phase 4: E2E Tests (Days 9-10)

**Day 9: Training Workflows**
- test_qlora_workflow.py
- test_training.py

**Day 10: Export Workflows**
- test_saving.py
- test_gguf_export.py

### Phase 5: Polish (Days 11-12)

**Day 11: Coverage and Reporting**
- Add coverage thresholds
- Set up PR comments
- Create test documentation

**Day 12: Review and Launch**
- Run full test suite
- Fix flaky tests
- Enable for all PRs

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | CI running unit tests |
| 6 | 50%+ unit test coverage |
| 8 | GPU tests running |
| 10 | E2E tests complete |
| 12 | Full pipeline live |

---

## Backwards Compatibility

### Breaking Changes

None. Tests are additive.

### New Requirements for Contributors

- Tests must pass before merge
- New features should include tests
- Coverage must not decrease

### Migration

No migration needed. Existing code unchanged.

---

## Alternatives Considered

### Alternative 1: Manual Testing Only

Continue with manual testing.

**Rejected:**
- Regressions reach main
- Slow PR review
- Inconsistent quality

### Alternative 2: Unit Tests Only (No GPU)

Only run tests that don't need GPU.

**Rejected:**
- Can't test core functionality
- False confidence
- Misses real issues

### Alternative 3: External CI Service

Use CircleCI, Travis, or other service.

**Rejected:**
- Another service to manage
- GitHub Actions sufficient
- Cost concerns

### Alternative 4: Test on Every PR with GPU

Run GPU tests on every PR.

**Rejected:**
- Too expensive
- Too slow
- Use labels for opt-in

---

## Open Questions

1. **Which GPU for runners?**
   - T4 (cheap, available)
   - A10 (faster)
   - Multiple for matrix?

2. **How to handle flaky tests?**
   - Retry mechanism
   - Quarantine
   - Skip in CI

3. **Test data management?**
   - Check in small datasets
   - Download on demand
   - Mock everything

4. **Performance regression tests?**
   - Benchmark suite
   - Track over time
   - Alert on regression

---

## Success Criteria

- [ ] Unit tests run on all PRs
- [ ] GPU tests run on main and labeled PRs
- [ ] Coverage reporting enabled
- [ ] Coverage threshold set (60%)
- [ ] No regressions after implementation
- [ ] Documentation for running tests

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Automated tests | 0 | 100+ |
| PR test coverage | 0% | 60%+ |
| Regressions caught | 0 | Many |
| Time to detect bugs | Days/weeks | Minutes |

---

## Required Approvals

- [ ] Maintainer approval for CI costs
- [ ] Infrastructure access for GPU runner
- [ ] Coverage threshold agreement

---

## Rollback Strategy

1. Disable failing workflows in GitHub
2. Remove required checks from branch protection
3. Tests remain but don't block

---

## References

- GitHub Actions Documentation: https://docs.github.com/en/actions
- Pytest Documentation: https://docs.pytest.org/
- Codecov: https://codecov.io/
- Self-hosted Runners: https://docs.github.com/en/actions/hosting-your-own-runners

---

*Next: [RFC-0007: Fused LoRA Forward Pass](./RFC-0007-fused-lora-forward.md)*
