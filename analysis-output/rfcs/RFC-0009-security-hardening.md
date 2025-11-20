# RFC-0009: Security Hardening and Input Validation

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 10 person-days
**Priority:** P0 (Critical)

---

## Summary

Implement comprehensive security hardening including input validation, safe model loading, secure credential storage, and protection against common vulnerabilities to ensure Unsloth is production-ready and secure for enterprise deployment.

---

## Motivation

Current security gaps pose risks for production deployments:

1. **No input validation:** Model names/paths not sanitized (path traversal risk)
2. **Unsafe deserialization:** Pickle files loaded without verification
3. **Insecure credentials:** HF tokens in plaintext environment variables
4. **Missing integrity checks:** No checksum verification for downloads
5. **Command injection risk:** Subprocess calls without sanitization

### Current Vulnerabilities

```python
# From unsloth/models/loader.py - no validation
def from_pretrained(model_name, **kwargs):
    # model_name could be "../../../etc/passwd"
    model = AutoModelForCausalLM.from_pretrained(model_name)
```

```python
# From unsloth/save.py - subprocess without sanitization
def save_to_gguf(model, output_path):
    subprocess.run(f"python convert.py {output_path}", shell=True)  # Injection risk
```

### Impact

| Vulnerability | Severity | Exploitability |
|--------------|----------|----------------|
| Path traversal | High | Easy |
| Pickle deserialization | Critical | Medium |
| Command injection | Critical | Medium |
| Token exposure | Medium | Easy |
| Missing checksums | Medium | Easy |

---

## Detailed Design

### 1. Input Validation Framework

```python
# unsloth/security/validation.py
import os
import re
from pathlib import Path
from typing import Union

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass

class InputValidator:
    """Centralized input validation."""

    # Allowed patterns
    HF_MODEL_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_./\-]+$')

    @staticmethod
    def validate_model_name(model_name: str) -> str:
        """
        Validate and sanitize model name.

        Accepts:
        - HuggingFace format: "org/model-name"
        - Absolute paths to existing directories
        - Relative paths within allowed directories

        Raises:
            ValidationError: If model_name is invalid or suspicious
        """
        if not model_name or not isinstance(model_name, str):
            raise ValidationError("Model name must be a non-empty string")

        # Check for path traversal attempts
        if ".." in model_name or model_name.startswith("/"):
            # Only allow if it's an absolute path to existing directory
            path = Path(model_name).resolve()
            if not path.exists() or not path.is_dir():
                raise ValidationError(
                    f"Invalid model path: {model_name}. "
                    "Path traversal detected or directory doesn't exist."
                )
            return str(path)

        # Check if it's a HuggingFace model ID
        if "/" in model_name:
            if not InputValidator.HF_MODEL_PATTERN.match(model_name):
                raise ValidationError(
                    f"Invalid HuggingFace model ID: {model_name}. "
                    "Must match pattern: org/model-name"
                )

        # Check for suspicious characters
        if not InputValidator.SAFE_PATH_PATTERN.match(model_name):
            raise ValidationError(
                f"Model name contains invalid characters: {model_name}"
            )

        return model_name

    @staticmethod
    def validate_save_path(path: Union[str, Path]) -> Path:
        """
        Validate output path for saving.

        Ensures path is within allowed directories and doesn't
        overwrite critical system files.
        """
        path = Path(path).resolve()

        # Prevent overwriting system directories
        forbidden_paths = [
            Path("/etc"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/var"),
        ]

        for forbidden in forbidden_paths:
            if path.is_relative_to(forbidden):
                raise ValidationError(
                    f"Cannot write to system directory: {path}"
                )

        # Check parent directory is writable
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        if not os.access(parent, os.W_OK):
            raise ValidationError(
                f"Permission denied: cannot write to {parent}"
            )

        return path
```

### 2. Safe Model Loading

```python
# unsloth/security/safe_loading.py
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Set

class SafeUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes.
    """

    ALLOWED_MODULES: Set[str] = {
        'torch',
        'numpy',
        'collections',
        'transformers.models',
        'peft',
    }

    def find_class(self, module, name):
        # Only allow specific modules
        module_root = module.split('.')[0]
        if module_root not in self.ALLOWED_MODULES:
            raise pickle.UnpicklingError(
                f"Global '{module}.{name}' is forbidden"
            )
        return super().find_class(module, name)

def safe_load_pickle(file_path: Path) -> object:
    """
    Safely load a pickle file with restricted classes.
    """
    with open(file_path, 'rb') as f:
        return SafeUnpickler(f).load()

def verify_model_checksum(
    file_path: Path,
    expected_sha256: Optional[str] = None,
) -> bool:
    """
    Verify model file integrity with SHA256 checksum.

    Args:
        file_path: Path to model file
        expected_sha256: Expected SHA256 hash (hex string)

    Returns:
        True if checksum matches or no expected hash provided

    Raises:
        SecurityError: If checksum doesn't match
    """
    if not expected_sha256:
        return True

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()

    if actual_hash != expected_sha256:
        raise SecurityError(
            f"Checksum mismatch for {file_path}\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {actual_hash}\n"
            "File may be corrupted or tampered with."
        )

    return True
```

### 3. Secure Credential Storage

```python
# unsloth/security/credentials.py
import keyring
import os
from typing import Optional

class CredentialManager:
    """
    Secure credential storage using system keyring.
    Falls back to environment variables with warning.
    """

    SERVICE_NAME = "unsloth"

    @staticmethod
    def store_token(token: str, username: str = "huggingface") -> None:
        """
        Store API token securely in system keyring.

        Example:
            >>> CredentialManager.store_token("hf_...")
        """
        try:
            keyring.set_password(
                CredentialManager.SERVICE_NAME,
                username,
                token
            )
        except Exception as e:
            raise SecurityError(
                f"Failed to store token securely: {e}\n"
                "Install keyring: pip install keyring"
            )

    @staticmethod
    def get_token(username: str = "huggingface") -> Optional[str]:
        """
        Retrieve API token from secure storage.
        Falls back to HF_TOKEN environment variable with warning.
        """
        # Try keyring first
        try:
            token = keyring.get_password(
                CredentialManager.SERVICE_NAME,
                username
            )
            if token:
                return token
        except Exception:
            pass

        # Fall back to environment variable
        token = os.environ.get("HF_TOKEN")
        if token:
            import warnings
            warnings.warn(
                "Using HF_TOKEN from environment variable. "
                "Consider using secure storage: "
                "CredentialManager.store_token(token)",
                SecurityWarning
            )

        return token

    @staticmethod
    def remove_token(username: str = "huggingface") -> None:
        """Remove stored token."""
        try:
            keyring.delete_password(
                CredentialManager.SERVICE_NAME,
                username
            )
        except Exception:
            pass
```

### 4. Safe Subprocess Execution

```python
# unsloth/security/subprocess_utils.py
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional

def safe_run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """
    Safely run a command without shell injection risk.

    Args:
        command: List of command parts (NOT shell string)
        cwd: Working directory
        timeout: Command timeout in seconds

    Example:
        >>> safe_run_command(["python", "convert.py", str(path)])

    Raises:
        SecurityError: If command is unsafe
        subprocess.TimeoutExpired: If timeout exceeded
    """
    # Validate command parts
    if not command or not isinstance(command, list):
        raise SecurityError("Command must be a non-empty list")

    for part in command:
        if not isinstance(part, str):
            raise SecurityError("All command parts must be strings")

    # Never use shell=True
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=True,
        text=True,
    )

    return result
```

### 5. Security Configuration

```python
# unsloth/security/config.py
from dataclasses import dataclass
from typing import Set

@dataclass
class SecurityConfig:
    """Security configuration for Unsloth."""

    # Input validation
    enable_path_validation: bool = True
    allow_relative_paths: bool = False

    # Model loading
    verify_checksums: bool = True
    use_safe_unpickler: bool = True
    allowed_pickle_modules: Set[str] = None

    # Credentials
    use_secure_storage: bool = True
    warn_on_env_credentials: bool = True

    # Subprocess
    subprocess_timeout: int = 300

    def __post_init__(self):
        if self.allowed_pickle_modules is None:
            self.allowed_pickle_modules = {
                'torch', 'numpy', 'collections',
                'transformers', 'peft'
            }
```

---

## Example Usage

### Secure Model Loading

```python
from unsloth import FastLanguageModel
from unsloth.security import SecurityConfig

# Enable all security features (default)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",  # Validated
    verify_checksum=True,           # Verify integrity
)

# Attempting path traversal raises error
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        "../../etc/passwd"  # Blocked!
    )
except ValidationError as e:
    print(f"Blocked: {e}")
```

### Secure Credential Storage

```python
from unsloth.security import CredentialManager

# Store token securely (one time)
CredentialManager.store_token("hf_your_token_here")

# Token retrieved automatically from keyring
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3-8B-private"  # Uses stored token
)
```

### Safe Model Saving

```python
from unsloth import save_to_gguf
from unsloth.security import validate_save_path

# Path validation
output_path = validate_save_path("./models/output.gguf")

# Safe subprocess execution (no shell injection)
save_to_gguf(model, tokenizer, output_path)
```

---

## Implementation Plan

### Phase 1: Input Validation (Days 1-3)

**Day 1: Validation Framework**
- Create `unsloth/security/` module
- Implement `InputValidator` class
- Add model name validation
- Add path validation

**Day 2: Integration**
- Add validation to `from_pretrained()`
- Add validation to save functions
- Write unit tests

**Day 3: Testing**
- Test with malicious inputs
- Test with edge cases
- Document validation rules

### Phase 2: Safe Loading (Days 4-6)

**Day 4: Safe Unpickler**
- Implement `SafeUnpickler`
- Define allowed modules
- Add checksum verification

**Day 5: Integration**
- Replace `pickle.load` calls
- Add checksum verification to downloads
- Handle HuggingFace model checksums

**Day 6: Testing**
- Test with restricted pickle files
- Test checksum verification
- Performance testing

### Phase 3: Credentials (Days 7-8)

**Day 7: Credential Manager**
- Implement `CredentialManager`
- Add keyring integration
- Add migration from env vars

**Day 8: Integration**
- Update token retrieval
- Add CLI for token management
- Document secure storage

### Phase 4: Subprocess (Days 9-10)

**Day 9: Safe Subprocess**
- Implement `safe_run_command`
- Replace all `subprocess` calls
- Remove `shell=True` usage

**Day 10: Testing & Documentation**
- Security testing
- Penetration testing
- Security documentation
- Security audit

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | Input validation complete |
| 6 | Safe loading complete |
| 8 | Secure credentials complete |
| 10 | Full security hardening |

---

## Backwards Compatibility

### Breaking Changes

**Potential issues:**
1. Path validation may reject previously-accepted paths
2. Pickle loading restrictions may fail for custom models
3. Keyring requires additional dependency

### Migration

```python
# Old (unsafe)
model, tokenizer = FastLanguageModel.from_pretrained("../models/custom")

# New (safe) - use absolute path
from pathlib import Path
model_path = Path("../models/custom").resolve()
model, tokenizer = FastLanguageModel.from_pretrained(str(model_path))
```

### Opt-out (not recommended)

```python
# Disable security features (NOT RECOMMENDED)
from unsloth.security import SecurityConfig

SecurityConfig.enable_path_validation = False  # Unsafe!
```

---

## Alternatives Considered

### Alternative 1: Third-Party Security Library

Use existing security libraries like `safety` or `bandit`.

**Rejected:**
- Generic, not ML-specific
- Doesn't address our specific vulnerabilities
- Still need custom implementation

### Alternative 2: Sandbox Execution

Run model loading in sandboxed environment.

**Rejected:**
- Complex to implement
- Performance overhead
- Doesn't solve all issues

### Alternative 3: Signature Verification

Require signed models only.

**Considered for future:**
- Good addition to checksums
- Requires key management
- Not all HF models are signed

---

## Open Questions

1. **Should we require checksums for all downloads?**
   - Pro: Maximum security
   - Con: HF API doesn't always provide checksums

2. **How to handle custom pickle classes?**
   - Allowlist configuration?
   - User-provided whitelist?

3. **Should we add rate limiting for API calls?**
   - Prevent abuse
   - Adds complexity

4. **Support for hardware security modules (HSM)?**
   - Enterprise requirement
   - Significant complexity

---

## Success Criteria

- [ ] 0 path traversal vulnerabilities
- [ ] All pickle loading uses SafeUnpickler
- [ ] Credentials stored securely by default
- [ ] All subprocess calls use safe_run_command
- [ ] Checksum verification for all downloads
- [ ] Security audit passes
- [ ] No breaking changes for standard usage

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Path validation | 0% | 100% |
| Safe pickle usage | 0% | 100% |
| Secure credential storage | 0% | 100% |
| Command injection risk | High | None |
| Security score (audit) | C | A |

---

## Required Approvals

- [ ] Security review
- [ ] Penetration testing
- [ ] Privacy compliance review
- [ ] Maintainer approval

---

## Rollback Strategy

1. Security features can be disabled individually
2. Environment variable to disable all security
3. No data format changes

---

## References

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Python Security Best Practices: https://python.readthedocs.io/en/stable/library/security_warnings.html
- Pickle Security: https://davidhamann.de/2020/04/05/exploiting-python-pickle/
- Keyring Library: https://pypi.org/project/keyring/

---

*Next: [RFC-0010: Automated Performance Benchmark Suite](./RFC-0010-benchmark-suite.md)*
