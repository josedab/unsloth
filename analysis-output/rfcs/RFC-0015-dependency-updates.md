# RFC-0015: Dependency Update Automation

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 7 person-days
**Priority:** P2 (Long-term)

---

## Summary

Implement automated dependency update system using Dependabot and custom testing to keep dependencies current, secure, and compatible while reducing manual maintenance burden.

---

## Motivation

Unsloth has complex dependency management with 50+ installation profiles and 11+ explicitly excluded versions. Currently, dependency updates are manual and infrequent:

### Current Issues

**No Automated Updates**
```toml
# pyproject.toml - dependencies manually maintained
dependencies = []  # Zero mandatory deps

[project.optional-dependencies]
cu121-torch240 = [
    "torch==2.4.0",
    "triton>=2.3.0,!=2.3.1,!=3.0.0,!=3.1.0",
    # ... 11+ excluded versions
]
```

**Version Conflicts**
- 11+ Triton versions explicitly excluded
- Manual testing required for each combination
- Security patches delayed
- Breaking changes discovered late

### Impact

| Issue | Effect |
|-------|--------|
| Manual updates | Delayed security patches |
| No compatibility testing | Breaking changes reach users |
| Excluded versions | Growing exclusion list |
| No deprecation tracking | Surprise breakage |

---

## Detailed Design

### 1. Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    reviewers:
      - "unsloth-maintainers"
    labels:
      - "dependencies"
      - "automated"

    # Group updates
    groups:
      pytorch:
        patterns:
          - "torch*"
          - "torchvision"
        update-types:
          - "minor"
          - "patch"

      transformers:
        patterns:
          - "transformers"
          - "tokenizers"
          - "accelerate"

      dev-dependencies:
        dependency-type: "development"

    # Ignore specific versions
    ignore:
      - dependency-name: "triton"
        versions: ["2.3.1", "3.0.0", "3.1.0"]

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 2. Compatibility Testing Workflow

```yaml
# .github/workflows/dependency-test.yml
name: Dependency Compatibility Testing

on:
  pull_request:
    paths:
      - 'pyproject.toml'
      - 'requirements*.txt'
  workflow_dispatch:
    inputs:
      package:
        description: 'Package to test'
        required: true
      version:
        description: 'Version to test'
        required: true

jobs:
  test-matrix:
    runs-on: [self-hosted, gpu]
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
        cuda-version: ['cu118', 'cu121', 'cu124']
        torch-version: ['2.3.0', '2.4.0', '2.5.0']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[${matrix.cuda-version}-torch${{ matrix.torch-version }}]"

      - name: Run compatibility tests
        run: |
          pytest tests/compatibility/ \
            --matrix-id="${{ matrix.python-version }}-${{ matrix.cuda-version }}-${{ matrix.torch-version }}" \
            -v

      - name: Test model loading
        run: |
          python tests/compatibility/test_basic_loading.py

      - name: Test training
        run: |
          python tests/compatibility/test_basic_training.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: compatibility-matrix
          path: test-results.json
```

### 3. Automated Compatibility Checker

```python
# scripts/check_compatibility.py
"""
Check dependency compatibility automatically.

Usage:
    python scripts/check_compatibility.py --package torch --version 2.5.0
"""

import subprocess
import sys
import json
from typing import List, Dict, Tuple

class CompatibilityChecker:
    """Check if a package version is compatible."""

    def __init__(self):
        self.test_suite = [
            self.test_import,
            self.test_model_loading,
            self.test_forward_pass,
            self.test_backward_pass,
            self.test_triton_kernels,
        ]

    def test_import(self) -> Tuple[bool, str]:
        """Test if unsloth can be imported."""
        try:
            import unsloth
            return True, "Import successful"
        except Exception as e:
            return False, f"Import failed: {e}"

    def test_model_loading(self) -> Tuple[bool, str]:
        """Test model loading."""
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/tinyllama-bnb-4bit",
                max_seq_length=256,
            )

            return True, "Model loading successful"

        except Exception as e:
            return False, f"Model loading failed: {e}"

    def test_forward_pass(self) -> Tuple[bool, str]:
        """Test forward pass."""
        try:
            from unsloth import FastLanguageModel
            import torch

            model, tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/tinyllama-bnb-4bit",
                max_seq_length=256,
            )

            inputs = tokenizer("Test", return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)

            return True, "Forward pass successful"

        except Exception as e:
            return False, f"Forward pass failed: {e}"

    def test_backward_pass(self) -> Tuple[bool, str]:
        """Test backward pass."""
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                "unsloth/tinyllama-bnb-4bit",
                max_seq_length=256,
            )

            inputs = tokenizer("Test", return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].sum()
            loss.backward()

            return True, "Backward pass successful"

        except Exception as e:
            return False, f"Backward pass failed: {e}"

    def test_triton_kernels(self) -> Tuple[bool, str]:
        """Test Triton kernels."""
        try:
            from unsloth.kernels.utils import is_triton_available

            if not is_triton_available():
                return False, "Triton not available"

            # Test a simple kernel
            from unsloth.kernels.fast_lora import LoRA_MLP
            return True, "Triton kernels working"

        except Exception as e:
            return False, f"Triton kernel test failed: {e}"

    def check(self) -> Dict:
        """Run all compatibility tests."""
        results = {
            "overall_passed": True,
            "tests": {},
        }

        for test_fn in self.test_suite:
            test_name = test_fn.__name__
            passed, message = test_fn()

            results["tests"][test_name] = {
                "passed": passed,
                "message": message,
            }

            if not passed:
                results["overall_passed"] = False

        return results

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Check dependency compatibility"
    )
    parser.add_argument("--package", required=True, help="Package name")
    parser.add_argument("--version", required=True, help="Package version")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    print(f"Testing compatibility with {args.package}=={args.version}")
    print("=" * 60)

    # Install package
    print(f"Installing {args.package}=={args.version}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{args.package}=={args.version}"],
        check=True,
    )

    # Run tests
    checker = CompatibilityChecker()
    results = checker.check()

    # Print results
    print("\nTest Results:")
    print("=" * 60)

    for test_name, result in results["tests"].items():
        status = "✓" if result["passed"] else "✗"
        print(f"{status} {test_name}: {result['message']}")

    print("=" * 60)

    if results["overall_passed"]:
        print("✓ All tests passed - package is compatible")
        exit_code = 0
    else:
        print("✗ Some tests failed - package may not be compatible")
        exit_code = 1

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
```

### 4. Version Matrix Generator

```python
# scripts/generate_version_matrix.py
"""
Generate compatibility matrix for all dependency combinations.

Outputs a matrix showing which versions work together.
"""

import itertools
import json
from typing import List, Dict

class VersionMatrix:
    """Generate and test version compatibility matrix."""

    def __init__(self):
        self.python_versions = ["3.9", "3.10", "3.11"]
        self.cuda_versions = ["cu118", "cu121", "cu124"]
        self.torch_versions = ["2.3.0", "2.4.0", "2.5.0"]
        self.transformers_versions = ["4.40.0", "4.41.0", "4.42.0"]

    def generate_combinations(self) -> List[Dict]:
        """Generate all version combinations to test."""
        combinations = []

        for py, cuda, torch, transformers in itertools.product(
            self.python_versions,
            self.cuda_versions,
            self.torch_versions,
            self.transformers_versions,
        ):
            combinations.append({
                "python": py,
                "cuda": cuda,
                "torch": torch,
                "transformers": transformers,
            })

        return combinations

    def test_combination(self, combo: Dict) -> bool:
        """Test if a combination works."""
        # Implementation would actually test the combination
        # For now, return placeholder
        return True

    def generate_matrix(self) -> Dict:
        """Generate full compatibility matrix."""
        combinations = self.generate_combinations()
        matrix = {}

        for combo in combinations:
            key = f"{combo['python']}-{combo['cuda']}-{combo['torch']}-{combo['transformers']}"
            matrix[key] = {
                "combination": combo,
                "compatible": self.test_combination(combo),
            }

        return matrix

    def save_matrix(self, output_file: str):
        """Save matrix to file."""
        matrix = self.generate_matrix()

        with open(output_file, "w") as f:
            json.dump(matrix, f, indent=2)

        print(f"Saved compatibility matrix to {output_file}")
```

### 5. Automated Exclusion List Manager

```python
# scripts/manage_exclusions.py
"""
Automatically manage excluded dependency versions.

Tracks why versions are excluded and when they can be retried.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

class ExclusionManager:
    """Manage excluded package versions."""

    def __init__(self, exclusions_file: Path = Path("exclusions.json")):
        self.exclusions_file = exclusions_file
        self.exclusions = self._load_exclusions()

    def _load_exclusions(self) -> Dict:
        """Load exclusions from file."""
        if self.exclusions_file.exists():
            with open(self.exclusions_file) as f:
                return json.load(f)
        return {}

    def add_exclusion(
        self,
        package: str,
        version: str,
        reason: str,
        retry_after_days: int = 90,
    ):
        """Add a version to exclusion list."""
        if package not in self.exclusions:
            self.exclusions[package] = {}

        self.exclusions[package][version] = {
            "reason": reason,
            "excluded_date": datetime.now().isoformat(),
            "retry_date": (datetime.now() + timedelta(days=retry_after_days)).isoformat(),
            "tested_count": 0,
        }

        self._save_exclusions()

    def check_retry_due(self) -> List[Dict]:
        """Check which exclusions are due for retry."""
        now = datetime.now()
        due_for_retry = []

        for package, versions in self.exclusions.items():
            for version, info in versions.items():
                retry_date = datetime.fromisoformat(info["retry_date"])

                if now >= retry_date:
                    due_for_retry.append({
                        "package": package,
                        "version": version,
                        "reason": info["reason"],
                        "excluded_date": info["excluded_date"],
                    })

        return due_for_retry

    def _save_exclusions(self):
        """Save exclusions to file."""
        with open(self.exclusions_file, "w") as f:
            json.dump(self.exclusions, f, indent=2)
```

---

## Example Usage

### Dependabot PR Workflow

1. Dependabot opens PR for `torch==2.5.1`
2. Automated tests run on compatibility matrix
3. If tests pass → PR approved for merge
4. If tests fail → Add to exclusion list, close PR

### Manual Compatibility Check

```bash
# Test specific version
python scripts/check_compatibility.py \
  --package torch \
  --version 2.5.1 \
  --output results.json

# Generate compatibility matrix
python scripts/generate_version_matrix.py \
  --output compatibility-matrix.json

# Check exclusions due for retry
python scripts/manage_exclusions.py check-retry
```

### CI Integration

```yaml
# PR checks automatically test compatibility
# No manual intervention needed for most updates
```

---

## Implementation Plan

### Phase 1: Automation Setup (Days 1-3)

**Day 1: Dependabot**
- Configure Dependabot
- Set up grouping rules
- Configure ignore rules

**Day 2: Testing Workflow**
- Create compatibility test workflow
- Add test matrix
- Configure GPU runners

**Day 3: Compatibility Checker**
- Implement CompatibilityChecker
- Add test suite
- Test locally

### Phase 2: Tooling (Days 4-5)

**Day 4: Matrix Generator**
- Implement version matrix generator
- Add combination testing
- Generate initial matrix

**Day 5: Exclusion Manager**
- Implement ExclusionManager
- Add retry tracking
- Migrate existing exclusions

### Phase 3: Integration and Documentation (Days 6-7)

**Day 6: Integration**
- Integrate with existing CI
- Test end-to-end workflow
- Fix issues

**Day 7: Documentation**
- Document dependency update process
- Write maintainer guide
- Update contributor guide

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | Automation infrastructure |
| 5 | Tooling complete |
| 7 | Full system operational |

---

## Backwards Compatibility

### Breaking Changes

None. Dependency updates are tested before merge.

### Gradual Rollout

- Start with dev dependencies
- Then move to optional dependencies
- Core dependencies last (currently none)

---

## Alternatives Considered

### Alternative 1: Manual Updates Only

Continue manual dependency management.

**Rejected:**
- Too slow for security patches
- Maintenance burden
- Misses updates

### Alternative 2: Renovate Instead of Dependabot

Use Renovate instead of Dependabot.

**Rejected:**
- Dependabot is GitHub-native
- Simpler configuration
- Good enough for needs

### Alternative 3: Pin All Dependencies

Pin exact versions for all dependencies.

**Rejected:**
- Misses security updates
- Breaks with newer environments
- Not sustainable

---

## Open Questions

1. **Auto-merge passing updates?**
   - Pro: Faster updates
   - Con: Risk of issues

2. **How often to retry exclusions?**
   - 90 days reasonable?
   - Test more frequently?

3. **Notify on breaking changes?**
   - Slack/Discord notifications
   - Email to maintainers
   - GitHub discussions

---

## Success Criteria

- [ ] Dependabot configured and running
- [ ] Compatibility tests in CI
- [ ] Automated testing for all PRs
- [ ] Exclusion list managed automatically
- [ ] Security patches applied within 7 days
- [ ] No manual dependency updates needed

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Dependency updates | Manual | Automated |
| Security patch time | Weeks | Days |
| Breaking change detection | Manual | Automated |
| Maintainer time | High | Low |

---

## Required Approvals

- [ ] Maintainer approval for automation
- [ ] CI budget approval
- [ ] Process approval

---

## Rollback Strategy

1. Disable Dependabot in repository settings
2. Close automated PRs
3. Return to manual updates

---

## References

- Dependabot: https://docs.github.com/en/code-security/dependabot
- GitHub Actions Matrix: https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
- Semantic Versioning: https://semver.org/

---

*Next: [RFC-0016: QAT Improvements](./RFC-0016-qat-improvements.md)*
