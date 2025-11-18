# Unsloth Development and Deployment Workflow Analysis

## 1. BUILD SYSTEM

### Build Tool
- **Primary Tool**: setuptools (v80.9.0)
- **Build Backend**: setuptools.build_meta
- **Version Control Tool**: setuptools-scm (v9.2.0) - for version management

### Version Management
- **Version Source**: Defined in `unsloth/models/_utils.py`
- **Current Version**: 2025.11.3
- **Version Strategy**: Uses date-based versioning (YYYY.MM.X format)
- **Dynamic Versioning**: Version is extracted at build time from the source file attribute:
  ```toml
  [tool.setuptools.dynamic]
  version = {attr = "unsloth.models._utils.__version__"}
  ```

### Package Configuration
- **Package Name**: unsloth
- **Python Support**: >=3.9,<3.14
- **License**: Apache-2.0
- **Homepage**: http://www.unsloth.ai
- **Repository**: https://github.com/unslothai/unsloth
- **Documentation**: https://github.com/unslothai/unsloth

### Packaging Format
- Distribution via wheel (.whl) and source distribution (sdist)
- No entry points defined (library, not CLI tool)
- Package data handling: include-package-data = false

### Dependency Management

**Core Dependencies**:
- pytorch (implicit)
- transformers >= 4.51.3 (with version exclusions)
- peft >= 0.7.1
- trl >= 0.18.2, <= 0.24.0
- huggingface_hub >= 0.34.0
- accelerate >= 0.34.1
- datasets
- numpy, tqdm, psutil

**Optional Dependencies Groups**:
1. **triton**: Triton >= 3.0.0 (Linux) or triton-windows (Windows)
2. **huggingface**: Full HuggingFace ecosystem
3. **windows**: Windows-specific with bitsandbytes
4. **base**: All huggingface dependencies
5. **CUDA-specific variants**: cu118, cu121, cu124, cu126, cu128, cu130 with various torch versions
   - Includes conditional xformers builds for different GPU architectures
   - Support for PyTorch 2.1.1 through 2.9.0

---

## 2. CI/CD PIPELINE

### GitHub Actions Workflows

#### Pre-commit Checks (`pre-commit.yml`)
- **Trigger**: On push to main, pull requests
- **Runner**: ubuntu-latest
- **Python**: 3.11
- **Tasks**:
  1. Code style and lint checks via pre-commit framework
  2. Runs all configured pre-commit hooks
  3. Validates formatting and code quality

#### Stale Issue Management (`stale.yml`)
- **Schedule**: Daily at 5:30 UTC
- **Purpose**: Inactivity management for GitHub issues
- **Configuration**:
  - Marks issues as "inactive" after 9999 days (effectively never auto-closes)
  - Posts community engagement message
  - Directs users to Reddit (/r/unsloth) and Discord communities

### Automated Checks on PRs
- Pre-commit hooks run automatically
- Code formatting validation via Ruff
- Custom kwarg spacing enforcement
- No dedicated unit test CI/CD (tests are local development only)

---

## 3. PRE-COMMIT HOOKS & CODE QUALITY

### Pre-commit Configuration (`.pre-commit-config.yaml`)

#### Ruff Linter & Formatter
- **Tool**: ruff v0.14.5 from astral-sh/ruff-pre-commit
- **Hook Configuration**:
  - Auto-fixes violations: `--fix`
  - Exits with non-zero on fixes: `--exit-non-zero-on-fix`

#### Custom Ruff Format Hook
- **Script**: `scripts/run_ruff_format.py`
- **Dependencies**: ruff==0.6.9
- **Purpose**: Run ruff format with custom kwarg spacing enforcement
- **Execution Chain**:
  1. First: `ruff format` for standard formatting
  2. Then: `enforce_kwargs_spacing.py` for custom spacing rules

#### Custom Kwargs Spacing Tool (`enforce_kwargs_spacing.py`)
Enforces two transformations:
1. **Spacing around `=` in keyword arguments**:
   - Adds space before `=` if missing
   - Adds space after `=` if missing
   - Example: `foo(a=1)` → `foo(a = 1)`

2. **Redundant Pass Statement Removal**:
   - Removes `pass` statements in blocks with other executable code
   - Uses AST parsing for accurate detection
   - Preserves `pass` in empty blocks (where it's required)

### Ruff Configuration (`pyproject.toml`)

#### Linting Rules
```toml
[tool.ruff.lint]
select = ["E9", "F63", "F7", "F82"]  # Critical errors only
ignore = [
    "E402", "E722", "F403", "F405",  # Module level imports
    "F811", "F821", "F841", "F401",  # Shadowing, undefined, unused
    "E731", "E741",                   # Lambda assignment, ambiguous names
    "F601", "E712",                   # Format strings, comparison to True/False
]
```

#### Formatting Rules
```toml
[tool.ruff]
target-version = "py311"
force-exclude = true
extend-exclude = [
    "*chat_templates.py",
    "*ollama_template_mappers.py",
    "*_auto_install.py",
    "*mapper.py",
]
```

---

## 4. DEVELOPMENT WORKFLOW

### Installation for Development

**Development Setup** (editable install):
```bash
pip install -e ".[huggingface]"
```

**With GPU Support** (CUDA 12.1):
```bash
pip install -e ".[huggingface,cu121-ampere,colab-no-deps]"
```

**Dependencies**:
- Pre-requisites: Python 3.9-3.13, PyTorch
- Development typically requires GPU (NVIDIA CUDA 7.0+)
- Supported GPUs: V100, T4, Titan V, RTX 20/30/40x, A100, H100, L40, etc.

### Testing Framework

#### Test Organization
- **Location**: `/tests/` directory
- **Test Types**:
  - Unit tests: Model registry, utilities
  - Integration tests: QLoRA training and merging
  - Saving/Loading tests: Perplexity validation, Hub uploads
  - MOE-specific tests: Qwen3, Llama4 MoE kernels
  - Quantization tests: QAT validation

#### Example Test Structure
```python
# Using pytest with parametrization
import pytest

@pytest.mark.parametrize("model_test_param", TestParams, ids=lambda p: p.name)
def test_model_registration(model_test_param):
    # Test implementation
    pass
```

#### Running Tests
- **Framework**: pytest (no explicit configuration in pyproject.toml)
- **Test execution**: `pytest tests/`
- **Typical workflow**:
  1. Run locally before PR
  2. Tests excluded from package distribution (`exclude = ["tests*"]`)

### Code Contribution Process

#### Submitting Issues
1. **Search First**: Check existing issues on GitHub
2. **Provide Details**:
   - Platform: Google Colab, Kaggle, local, cloud
   - Environment: OS, Python version, GPU model
   - Reproduction: Minimal code snippet
   - Logs: Traceback and error messages
   - Screenshots: Visual issues

#### Issue Templates
1. **Bug Reports** (`ISSUE_TEMPLATE/bug---issue.md`):
   - Ask about updates
   - Platform specifics
   - GPU and software versions
   - Training components used (SFTTrainer, GRPOTrainer, etc.)
   - Minimal reproduction code

2. **Feature Requests** (`ISSUE_TEMPLATE/feature-request.md`):
   - Suggests trying FastModel with custom models
   - Covers new model support requests

#### Pull Request Process
1. Create feature branch
2. Implement changes
3. Run pre-commit hooks locally:
   ```bash
   pre-commit run --all-files
   ```
4. Commit with clear messages
5. Push and create PR
6. Address CI feedback
7. Code review by maintainers
8. Merge to main

#### Contribution Types
- Code fixes for reported bugs
- New feature implementations
- Documentation improvements (docs.unsloth.ai)
- Community support (answering issues/discussions)
- Blog posts and social media sharing

#### Maintainers
- Daniel Han (danielhanchen@gmail.com)
- Michael Han (info@unsloth.ai)

---

## 5. RELEASE PROCESS

### Version Management Strategy
- **Format**: YYYY.MM.PATCH (e.g., 2025.11.3)
- **Source of Truth**: `unsloth/models/_utils.py`
- **Line Example**: `__version__ = "2025.11.3"`

### Release Workflow
1. **Version Bump**: Update `__version__` in `unsloth/models/_utils.py`
2. **Build**: setuptools creates wheel and sdist from pyproject.toml
3. **Distribution**: Uploaded to PyPI
   - Format: `.whl` (wheel) for binary distribution
   - Format: `.tar.gz` (sdist) for source distribution

### Supported Installation Methods
1. **PyPI**: `pip install unsloth`
2. **From Source**: `pip install -e ".[huggingface]"`
3. **Docker**: Official Unsloth Docker image available at `unsloth/unsloth`
4. **Platform-Specific**:
   - Windows: Requires PyTorch pre-installed
   - WSL: Supported with Linux libraries
   - Docker: Eliminates environment setup issues

### Configuration Management
- **Setup Tools Configuration**: Centralized in `pyproject.toml`
- **No setup.cfg**: Modern setuptools-only approach
- **No setup.py**: Entirely declarative configuration
- **No tox.ini**: No multi-environment testing automation

### Package Exclusions
During build, the following are excluded:
- `images*` - Documentation/marketing images
- `tests*` - Test suite excluded from distribution
- `kernels/moe*` - MOE kernel implementations

---

## 6. QUALITY ASSURANCE & STANDARDS

### Code Quality Standards

#### Style Rules
- Target Python 3.11 compatibility
- Keyword argument spacing enforcement: `func(a = 1)` format
- No redundant `pass` statements allowed
- Critical error checking (E9, F63, F7, F82)

#### Excluded Patterns (by file)
Files explicitly excluded from linting:
- `*chat_templates.py` - Large template files
- `*ollama_template_mappers.py` - External mappings
- `*_auto_install.py` - Installation scripts
- `*mapper.py` - Mapping utilities

### Pre-commit Workflow
1. Developer makes code changes
2. Runs `pre-commit run --all-files` locally
3. Hook automatically fixes:
   - Formatting violations
   - Kwarg spacing issues
   - Redundant pass statements
4. Developer reviews changes
5. Commits with fixes applied
6. CI validates via GitHub Actions

### Community Standards
- Code of Conduct enforcement
- Inclusive environment requirements
- Active Discord community (https://discord.com/invite/unsloth)
- Reddit community engagement (/r/unsloth)

---

## 7. DOCUMENTATION & SUPPORT

### Documentation Sources
- **Official Docs**: https://docs.unsloth.ai
- **GitHub Repo**: https://github.com/unslothai/unsloth
- **Blog**: https://unsloth.ai/blog
- **Notebooks**: Community Google Colab examples

### Community Channels
- Discord: https://discord.com/invite/unsloth
- Reddit: https://www.reddit.com/r/unsloth
- Twitter/X: https://twitter.com/unslothai

### Support Tiers
1. **GitHub Issues**: Technical support and feature requests
2. **Discord**: Real-time community discussion
3. **Reddit**: General questions and sharing
4. **Sponsorship**: Ko-fi support available

---

## SUMMARY

Unsloth follows a modern Python development workflow:

**Development**: 
- setuptools-based build system with dynamic versioning
- Date-based version scheme maintained in source
- Comprehensive optional dependencies for GPU/CUDA variants

**Quality Assurance**:
- Lightweight pre-commit hooks (Ruff + custom scripts)
- Focus on critical errors, not style perfection
- Custom kwarg spacing enforcement via AST manipulation
- No automated testing in CI/CD (local developer responsibility)

**Collaboration**:
- GitHub-based issue templates and workflows
- Stale issue management for community engagement
- Active community on Discord and Reddit
- Centralized documentation at docs.unsloth.ai

**Release Management**:
- Simple date-based versioning
- Direct PyPI distribution
- Multi-platform support (Linux, WSL, Windows, Docker)
- Extensive GPU support with variant installations

**Key Characteristics**:
- Performance-focused (ML inference and training optimization)
- Community-driven with active engagement channels
- GPU-centric with extensive CUDA version variants
- Minimal CI/CD automation (trusts developer testing)
- Clear separation of concerns (pre-commit, tests, releases)
