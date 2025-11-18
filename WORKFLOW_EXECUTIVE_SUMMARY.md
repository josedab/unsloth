# Unsloth Development & Deployment Workflow - Executive Summary

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Build Tool** | setuptools 80.9.0 (no setup.py) |
| **Version Strategy** | Date-based (YYYY.MM.PATCH), e.g., 2025.11.3 |
| **Version Source** | `unsloth/models/_utils.py` |
| **Python Support** | 3.9 - 3.13 |
| **CI/CD** | GitHub Actions (pre-commit only) |
| **Code Quality** | Ruff + Custom kwarg spacing |
| **Testing** | pytest (local dev responsibility) |
| **Distribution** | PyPI, Docker, GitHub source |
| **Maintainers** | Daniel Han, Michael Han |

---

## Key Findings

### 1. Build System - Modern & Declarative
- **setuptools 80.9.0** with `setuptools.build_meta` backend
- No `setup.py` or `setup.cfg` - everything in `pyproject.toml`
- Dynamic version from source attribute: `{attr = "unsloth.models._utils.__version__"}`
- Extensive optional dependencies for GPU/CUDA variants (cu118-cu130)
- Wheel and source distributions to PyPI

### 2. CI/CD - Minimal Automation
- **Only 2 GitHub Actions workflows:**
  - `pre-commit.yml`: Runs linting on push to main and PRs
  - `stale.yml`: Daily issue lifecycle management
- **No automated testing** - relies on developer responsibility
- **No build/publish automation** - releases are manual
- Pre-commit hooks run on Python 3.11 in Ubuntu environment

### 3. Code Quality - Focused & Custom
- **Ruff 0.14.5** for linting and formatting
- **Custom hooks** in `scripts/` for:
  - Kwarg spacing enforcement (`func(a = 1)` format)
  - Redundant pass statement removal
- **Lenient rules**: Only critical errors (E9, F63, F7, F82)
- Ignores ~15 style rules for developer flexibility
- Excludes 4 file patterns from linting

### 4. Development Workflow - GPU-Centric
- Development installation: `pip install -e ".[huggingface]"`
- Requires GPU (NVIDIA CUDA 7.0+) for effective development
- Testing via pytest (local developer runs tests, not CI)
- Issue templates for bugs and feature requests
- Clear contribution process documented in CONTRIBUTING.md

### 5. Testing - Developer-Driven
- **Framework**: pytest (no explicit config)
- **Location**: `/tests/` directory (6 test modules)
- **Types**: Unit, integration, quantization, MoE-specific
- **Not in CI**: Tests are developer responsibility
- **Not in package**: Tests excluded from distribution

### 6. Release Process - Simple & Manual
1. Edit version in `unsloth/models/_utils.py`
2. Run `python -m build` locally
3. Run `twine upload dist/*` to PyPI
4. Update Docker image separately

No automated version bumping, git tagging, or CI publishing.

### 7. Community & Support
- **Active channels**: Discord, Reddit, GitHub Issues
- **Documentation**: https://docs.unsloth.ai (external)
- **Sponsorship**: Ko-fi
- **Issue management**: Daily stale issue pinger

---

## Architecture Highlights

### Strengths
1. **Simplicity**: Minimal CI/CD reduces complexity and maintenance
2. **Flexibility**: Lenient code quality rules allow developer autonomy
3. **Performance-focused**: Custom hooks for specific performance patterns
4. **Community-driven**: Multiple engagement channels
5. **GPU-optimized**: Extensive CUDA variant support

### Tradeoffs
1. **Manual releases**: No CI automation, risk of human error
2. **Developer testing**: CI doesn't validate functionality
3. **No automated changelog**: Manual release notes needed
4. **Limited release tracking**: No git tags or automated version management
5. **Manual Docker updates**: Separate from code releases

---

## Developer Workflow Summary

### For Contributors
```
1. Fork repo
2. pip install -e ".[huggingface]"  (with GPU)
3. Make changes
4. pytest tests/  (optional, local)
5. pre-commit run --all-files  (required)
6. git commit -m "Clear message"
7. Create PR
8. Address CI feedback
9. Code review & merge
```

### For Maintainers (Release)
```
1. Update __version__ in unsloth/models/_utils.py
2. python -m build
3. twine upload dist/*
4. Update Docker image
5. GitHub/social media announcement
```

---

## Files to Know

**Build Configuration**:
- `/home/user/unsloth/pyproject.toml` - Build and project metadata
- `/home/user/unsloth/unsloth/models/_utils.py` - Version definition

**CI/CD**:
- `/home/user/unsloth/.github/workflows/pre-commit.yml` - Linting
- `/home/user/unsloth/.github/workflows/stale.yml` - Issue management

**Code Quality**:
- `/home/user/unsloth/.pre-commit-config.yaml` - Hook definitions
- `/home/user/unsloth/scripts/run_ruff_format.py` - Formatter orchestration
- `/home/user/unsloth/scripts/enforce_kwargs_spacing.py` - Custom formatter

**Contribution**:
- `/home/user/unsloth/CONTRIBUTING.md` - Guidelines
- `/home/user/unsloth/.github/ISSUE_TEMPLATE/` - Templates

**Testing**:
- `/home/user/unsloth/tests/` - Test suite

---

## Recommendations for Contributors

### Before Submitting PR
```bash
# Install development dependencies
pip install -e ".[huggingface]"

# Install pre-commit hook
pre-commit install

# Run pre-commit locally
pre-commit run --all-files

# Run tests
pytest tests/

# Commit when all pass
git commit -m "Descriptive message"
```

### Issue Reporting Best Practices
- Include: Python version, PyTorch version, GPU model, CUDA version
- For bugs: Minimal reproducible code snippet
- For features: Explain use case and benefit
- Check existing issues first

### Code Quality Standards
- Leverage Ruff auto-fixes (it fixes most issues)
- Ensure kwarg spacing (spaces around `=`)
- Remove redundant pass statements
- Test your changes locally with pytest
- Keep commits atomic and well-described

---

## Unique Characteristics

1. **No Conventional Testing in CI**: Unlike most projects, Unsloth doesn't run tests in CI. This works because:
   - Project is performance-critical (hardware testing needed)
   - Community-driven (crowd testing)
   - GPU requirements (expensive CI)

2. **Custom Kwarg Spacing**: Unusual style enforcement:
   - `func(a = 1)` instead of standard `func(a=1)`
   - Enforced via AST manipulation
   - Part of project identity

3. **Date-Based Versioning**: YYYY.MM.PATCH format:
   - Clear release timeline
   - Easy to identify old versions
   - Aligns with monthly release cadence

4. **Manual Release Management**: No automation:
   - Maintainer control
   - Batch multiple PRs before release
   - Risk of human error mitigated by small core team

5. **GPU-Centric Infrastructure**: 
   - All development targets NVIDIA GPUs
   - Extensive CUDA variant support
   - Installation varies by hardware

---

## Potential Improvements

1. **Add GitHub Release automation** - Create releases from git tags
2. **Implement test CI** - At least for critical paths
3. **Changelog automation** - Generate from conventional commits
4. **Dependency pinning policy** - Clear rules for version compatibility
5. **Docker image automation** - Build images on release

---

## Conclusion

Unsloth follows a **lightweight, community-driven development model** optimized for:
- **Performance**: Custom code quality, GPU-first design
- **Simplicity**: Minimal CI/CD, manual processes
- **Community**: Active engagement channels, issue templates
- **Flexibility**: Lenient code rules, developer autonomy

This approach works well for a performance-focused ML library with strong community support, though larger projects might require more automation and structured testing.

