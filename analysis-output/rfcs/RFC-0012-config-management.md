# RFC-0012: Centralized Configuration Management

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 7 person-days
**Priority:** P1 (Strategic)

---

## Summary

Implement centralized configuration system to replace scattered environment variables, hardcoded constants, and implicit defaults with a unified, validated, and well-documented configuration framework.

---

## Motivation

Unsloth's configuration is currently scattered across multiple sources:

**Environment Variables (scattered)**
```python
# From various files
os.environ.get("HF_TOKEN")
os.environ.get("UNSLOTH_DISABLE_TRITON")
os.environ.get("CUDA_VISIBLE_DEVICES")
# ... many more, undocumented
```

**Hardcoded Constants (in code)**
```python
# From unsloth/models/_utils.py
SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
MAX_SEQUENCE_LENGTH = 4096
DEFAULT_ROPE_SCALING = 1.0
```

**Function Parameters (implicit defaults)**
```python
def from_pretrained(
    model_name,
    max_seq_length=None,  # What's the actual default?
    dtype=None,            # How is this determined?
    load_in_4bit=True,     # When should this be False?
    ...
)
```

### Problems

| Issue | Impact |
|-------|--------|
| No single source of truth | Configuration scattered across codebase |
| Undocumented env vars | Users don't know what's available |
| No validation | Invalid configs cause runtime errors |
| No precedence rules | Unclear which config takes priority |
| Hard to test | Can't easily mock configurations |

### Current Pain Points

1. **Discoverability**: Users don't know what can be configured
2. **Validation**: Invalid values cause cryptic errors
3. **Documentation**: No comprehensive config reference
4. **Testing**: Hard to test different configurations
5. **Defaults**: Unclear default behavior

---

## Detailed Design

### Configuration Hierarchy

```python
# unsloth/config.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import os

@dataclass
class UnslothConfig:
    """
    Centralized Unsloth configuration.

    Configuration precedence (highest to lowest):
    1. Explicit parameters passed to functions
    2. Environment variables (UNSLOTH_*)
    3. Config file (~/.config/unsloth/config.toml)
    4. Defaults

    Environment variables:
        UNSLOTH_CACHE_DIR: Model cache directory
        UNSLOTH_HF_TOKEN: HuggingFace API token
        UNSLOTH_DEFAULT_DTYPE: Default dtype (float16/bfloat16/float32)
        UNSLOTH_DISABLE_TRITON: Disable Triton kernels (0/1)
        UNSLOTH_LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)
        UNSLOTH_MAX_SEQ_LENGTH: Default max sequence length
        UNSLOTH_ENABLE_FLASH_ATTN: Enable Flash Attention (0/1)

    Example:
        >>> from unsloth import get_config, set_config
        >>> config = get_config()
        >>> print(config.cache_dir)
        /home/user/.cache/unsloth

        >>> set_config(default_dtype="bfloat16")
        >>> config = get_config()
        >>> print(config.default_dtype)
        bfloat16
    """

    # Model Loading
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "unsloth"
    )
    max_seq_length: Optional[int] = None
    default_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None
    load_in_4bit: bool = True

    # Authentication
    hf_token: Optional[str] = None

    # Performance
    enable_flash_attn: bool = True
    enable_triton: bool = True
    num_workers: int = 4

    # Memory
    gradient_checkpointing: bool = True
    max_memory_per_gpu: Optional[int] = None  # GB

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[Path] = None

    # Training
    default_lora_r: int = 16
    default_lora_alpha: int = 16
    default_lora_dropout: float = 0.0

    # Advanced
    rope_scaling: Optional[float] = None
    trust_remote_code: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert strings to Path
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        if self.log_file and isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)

        # Validate dtype
        if self.default_dtype and self.default_dtype not in ["float16", "bfloat16", "float32"]:
            raise ValueError(
                f"Invalid dtype: {self.default_dtype}. "
                "Must be one of: float16, bfloat16, float32"
            )

        # Validate sequence length
        if self.max_seq_length and self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "UnslothConfig":
        """Load configuration from environment variables."""
        return cls(
            cache_dir=os.getenv("UNSLOTH_CACHE_DIR", cls.cache_dir),
            hf_token=os.getenv("UNSLOTH_HF_TOKEN"),
            default_dtype=os.getenv("UNSLOTH_DEFAULT_DTYPE"),
            enable_triton=os.getenv("UNSLOTH_DISABLE_TRITON", "0") != "1",
            log_level=os.getenv("UNSLOTH_LOG_LEVEL", "INFO"),
            max_seq_length=int(os.getenv("UNSLOTH_MAX_SEQ_LENGTH"))
                if os.getenv("UNSLOTH_MAX_SEQ_LENGTH") else None,
            enable_flash_attn=os.getenv("UNSLOTH_ENABLE_FLASH_ATTN", "1") == "1",
        )

    @classmethod
    def from_file(cls, path: Path) -> "UnslothConfig":
        """Load configuration from TOML file."""
        import tomli

        with open(path, "rb") as f:
            data = tomli.load(f)

        return cls(**data.get("unsloth", {}))

    def to_file(self, path: Path):
        """Save configuration to TOML file."""
        import tomli_w
        from dataclasses import asdict

        config_dict = {
            "unsloth": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(self).items()
                if v is not None
            }
        }

        with open(path, "wb") as f:
            tomli_w.dump(config_dict, f)
```

### Configuration Manager

```python
# unsloth/config.py (continued)

class ConfigManager:
    """Singleton configuration manager."""

    _instance: Optional[UnslothConfig] = None
    _config_file = Path.home() / ".config" / "unsloth" / "config.toml"

    @classmethod
    def get_config(cls) -> UnslothConfig:
        """Get current configuration."""
        if cls._instance is None:
            cls._instance = cls._load_config()
        return cls._instance

    @classmethod
    def set_config(cls, **kwargs):
        """Update configuration."""
        current = cls.get_config()
        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    @classmethod
    def reset_config(cls):
        """Reset to default configuration."""
        cls._instance = UnslothConfig()

    @classmethod
    def _load_config(cls) -> UnslothConfig:
        """
        Load configuration with precedence:
        1. Config file
        2. Environment variables
        3. Defaults
        """
        # Start with defaults
        config = UnslothConfig()

        # Override with config file if exists
        if cls._config_file.exists():
            file_config = UnslothConfig.from_file(cls._config_file)
            for key, value in file_config.__dict__.items():
                if value is not None:
                    setattr(config, key, value)

        # Override with environment variables
        env_config = UnslothConfig.from_env()
        for key, value in env_config.__dict__.items():
            if value is not None:
                setattr(config, key, value)

        return config

# Convenience functions
def get_config() -> UnslothConfig:
    """Get current Unsloth configuration."""
    return ConfigManager.get_config()

def set_config(**kwargs):
    """
    Update Unsloth configuration.

    Example:
        >>> set_config(default_dtype="bfloat16", enable_triton=False)
    """
    ConfigManager.set_config(**kwargs)

def reset_config():
    """Reset configuration to defaults."""
    ConfigManager.reset_config()
```

### CLI for Configuration

```python
# unsloth/cli/config.py
import click
from unsloth.config import get_config, set_config, ConfigManager

@click.group()
def config():
    """Manage Unsloth configuration."""
    pass

@config.command()
def show():
    """Show current configuration."""
    cfg = get_config()

    click.echo("Current Unsloth Configuration:")
    click.echo("=" * 50)

    for key, value in cfg.__dict__.items():
        click.echo(f"{key:25s} = {value}")

@config.command()
@click.argument("key")
@click.argument("value")
def set(key, value):
    """Set a configuration value."""
    try:
        # Try to parse value
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)

        set_config(**{key: value})
        click.echo(f"✓ Set {key} = {value}")

    except ValueError as e:
        click.echo(f"✗ Error: {e}", err=True)

@config.command()
@click.argument("key")
def get(key):
    """Get a configuration value."""
    cfg = get_config()

    if hasattr(cfg, key):
        value = getattr(cfg, key)
        click.echo(value)
    else:
        click.echo(f"✗ Unknown config key: {key}", err=True)

@config.command()
def init():
    """Initialize configuration file."""
    cfg = get_config()
    config_file = ConfigManager._config_file

    config_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_file(config_file)

    click.echo(f"✓ Created config file: {config_file}")

if __name__ == "__main__":
    config()
```

### Integration with Existing Code

```python
# unsloth/models/loader.py (updated)
from unsloth.config import get_config

def from_pretrained(
    model_name: str,
    max_seq_length: Optional[int] = None,
    dtype: Optional[str] = None,
    load_in_4bit: Optional[bool] = None,
    **kwargs,
):
    """
    Load model with Unsloth optimizations.

    Args are now optional and use config defaults if not specified.
    """
    config = get_config()

    # Use explicit parameters if provided, otherwise use config
    max_seq_length = max_seq_length or config.max_seq_length
    dtype = dtype or config.default_dtype
    load_in_4bit = load_in_4bit if load_in_4bit is not None else config.load_in_4bit

    # Use config for other settings
    if "token" not in kwargs and config.hf_token:
        kwargs["token"] = config.hf_token

    if "cache_dir" not in kwargs:
        kwargs["cache_dir"] = str(config.cache_dir)

    # Rest of loading logic...
```

---

## Example Usage

### Using Configuration

```python
from unsloth import FastLanguageModel, get_config, set_config

# Check current config
config = get_config()
print(f"Cache dir: {config.cache_dir}")
print(f"Default dtype: {config.default_dtype}")

# Update config
set_config(
    default_dtype="bfloat16",
    max_seq_length=4096,
    enable_triton=True,
)

# Now all models use these defaults
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit"
    # max_seq_length=4096 used automatically
    # dtype=bfloat16 used automatically
)
```

### Config File

```toml
# ~/.config/unsloth/config.toml
[unsloth]
cache_dir = "/data/models"
default_dtype = "bfloat16"
max_seq_length = 4096
hf_token = "hf_..."
enable_flash_attn = true
enable_triton = true
log_level = "INFO"
default_lora_r = 16
```

### Environment Variables

```bash
# Override config file with environment
export UNSLOTH_CACHE_DIR=/data/models
export UNSLOTH_DEFAULT_DTYPE=bfloat16
export UNSLOTH_HF_TOKEN=hf_...

python train.py  # Uses env vars
```

### CLI Usage

```bash
# Show current config
unsloth config show

# Set a value
unsloth config set default_dtype bfloat16
unsloth config set max_seq_length 4096

# Get a value
unsloth config get cache_dir

# Initialize config file
unsloth config init
```

---

## Implementation Plan

### Phase 1: Core Config System (Days 1-3)

**Day 1: Configuration Class**
- Implement UnslothConfig dataclass
- Add validation logic
- Add from_env() method

**Day 2: Config Manager**
- Implement ConfigManager singleton
- Add precedence logic
- Add config file loading (TOML)

**Day 3: Testing**
- Test validation
- Test precedence rules
- Test config file parsing

### Phase 2: Integration (Days 4-5)

**Day 4: Update Loaders**
- Update FastLanguageModel.from_pretrained()
- Update FastVisionModel.from_pretrained()
- Update save functions

**Day 5: Update Training**
- Update UnslothTrainer
- Update trainer arguments
- Ensure backward compatibility

### Phase 3: CLI and Documentation (Days 6-7)

**Day 6: CLI Implementation**
- Implement config CLI commands
- Add to main CLI
- Test CLI

**Day 7: Documentation**
- Document all config options
- Write migration guide
- Update README

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | Core config system |
| 5 | Integration complete |
| 7 | CLI and docs ready |

---

## Backwards Compatibility

### Breaking Changes

None. All existing code continues to work.

### Migration Path

```python
# Old way (still works)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype="float16",
)

# New way (optional)
from unsloth import set_config

set_config(max_seq_length=2048, default_dtype="float16")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit"
)
```

### Environment Variable Changes

Old env vars still work with deprecation warning:
```python
# Old (still works, shows warning)
os.environ["HF_TOKEN"] = "..."

# New (recommended)
os.environ["UNSLOTH_HF_TOKEN"] = "..."
```

---

## Alternatives Considered

### Alternative 1: Continue with Current Approach

Keep configuration scattered.

**Rejected:**
- Poor user experience
- Hard to document
- Difficult to test

### Alternative 2: Use Hydra/OmegaConf

Use existing config framework.

**Rejected:**
- Heavy dependency
- Overcomplicated for needs
- Different paradigm from HF

### Alternative 3: JSON Config Files

Use JSON instead of TOML.

**Rejected:**
- TOML more human-friendly
- Comments not supported in JSON
- TOML is Python standard (PEP 680)

---

## Open Questions

1. **Should config file be auto-created?**
   - Pro: Easier for users
   - Con: Clutters home directory

2. **Support multiple config profiles?**
   - E.g., dev, prod, test
   - Adds complexity
   - Nice to have

3. **Validate on load or on use?**
   - Load: Fail fast
   - Use: More flexible
   - Current: Validate on load

---

## Success Criteria

- [ ] All configuration centralized in UnslothConfig
- [ ] Config file support working
- [ ] Environment variable support working
- [ ] CLI for config management
- [ ] Comprehensive documentation
- [ ] No breaking changes

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Config locations | Scattered | 1 class |
| Documented config options | ~30% | 100% |
| Validation | None | Complete |
| User complaints about config | High | Low |

---

## Required Approvals

- [ ] API design review
- [ ] Documentation review
- [ ] User testing

---

## Rollback Strategy

1. Config system is opt-in
2. Can remove without breaking existing code
3. Existing parameter passing still works

---

## References

- Python dataclasses: https://docs.python.org/3/library/dataclasses.html
- TOML specification: https://toml.io/
- HuggingFace config: https://huggingface.co/docs/transformers/main_classes/configuration
- Click CLI: https://click.palletsprojects.com/

---

*Next: [RFC-0013: Enhanced Error Messages with Troubleshooting](./RFC-0013-error-messages.md)*
