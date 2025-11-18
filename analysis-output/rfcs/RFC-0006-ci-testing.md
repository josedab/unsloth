# RFC-0006: Automated CI Testing Pipeline

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 12 person-days
**Priority:** P1 (Strategic)

---

## Summary

Implement automated testing in CI/CD pipeline using GitHub Actions with GPU runners, achieving automated quality assurance for all PRs.

---

## Motivation

Currently, Unsloth has no automated testing in CI. Tests are run manually by developers, leading to:
- Regressions reaching main branch
- Inconsistent quality across PRs
- No coverage reporting
- Contributor uncertainty about test expectations

### Current CI

Only two workflows exist:
- Pre-commit linting
- Stale issue management

---

## Detailed Design

### Test Categories

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # Fast tests without GPU
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/unit -v --cov=unsloth

  # GPU tests (more expensive)
  integration-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: pytest tests/integration -v
```

### Test Structure

```
tests/
├── unit/                    # No GPU required
│   ├── test_registry.py
│   ├── test_mapper.py
│   ├── test_validation.py
│   └── test_config.py
├── integration/             # GPU required
│   ├── test_model_loading.py
│   ├── test_training.py
│   └── test_saving.py
└── e2e/                     # Full workflow tests
    ├── test_qlora_workflow.py
    └── test_gguf_export.py
```

### GPU Runner Options

1. **GitHub GPU runners** (when available)
2. **Self-hosted runners** with GPU
3. **Cloud GPU runners** (AWS, GCP, Lambda Labs)

---

## Implementation Plan

### Phase 1: Unit Tests (Days 1-4)
1. Create unit test structure
2. Mock GPU-dependent code
3. Add to CI pipeline
4. Target 80% coverage for non-GPU code

### Phase 2: Integration Tests (Days 5-8)
1. Set up GPU runner
2. Create integration test suite
3. Test model loading and training
4. Add to CI (PR optional, main required)

### Phase 3: Coverage & Reporting (Days 9-12)
1. Add coverage reporting
2. Set coverage thresholds
3. Add badge to README
4. Create test documentation

---

## Cost Considerations

| Option | Cost/Month | Pros | Cons |
|--------|-----------|------|------|
| Self-hosted | Hardware | Full control | Maintenance |
| Cloud spot | ~$50-100 | Flexible | Cold start |
| GitHub GPU | TBD | Integrated | Availability |

---

## Success Criteria

- [ ] All PRs run unit tests automatically
- [ ] GPU tests run on main branch
- [ ] Coverage reporting enabled
- [ ] No regressions in test suite

---

*Next: [RFC-0007: Fused LoRA Forward Pass](./RFC-0007-fused-lora-forward.md)*
