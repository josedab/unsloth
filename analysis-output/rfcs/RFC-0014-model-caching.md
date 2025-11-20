# RFC-0014: Intelligent Model Caching and Disk Management

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 10 person-days
**Priority:** P2 (Long-term)

---

## Summary

Implement intelligent model caching with automatic disk space management, cache eviction policies, and fast model loading to reduce download times and manage disk usage effectively.

---

## Motivation

Current model caching issues:

1. **No cache management**: Models accumulate indefinitely in `~/.cache/huggingface/hub/`
2. **No disk space monitoring**: Cache can fill disk without warning
3. **No eviction policy**: Old/unused models never removed
4. **Redundant downloads**: Same model downloaded multiple times for different quantizations
5. **Slow loading**: No optimization for repeated loads

### Current Problems

```bash
$ du -sh ~/.cache/huggingface/hub/
184G    ~/.cache/huggingface/hub/

# User doesn't know:
# - Which models are cached
# - Which models are used recently
# - How to clean up safely
# - When disk will be full
```

### Impact

| Issue | Effect |
|-------|--------|
| Unlimited cache growth | Fills disk unexpectedly |
| No cleanup tools | Manual deletion required |
| Redundant storage | Same model in multiple formats |
| Slow loading | No optimization for hot models |

---

## Detailed Design

### 1. Cache Manager

```python
# unsloth/cache/manager.py
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import json
import time
import shutil

@dataclass
class CachedModel:
    """Information about a cached model."""
    model_name: str
    path: Path
    size_mb: float
    last_accessed: float
    access_count: int
    quantization: Optional[str] = None

class CacheManager:
    """
    Manage model cache with intelligent eviction.

    Features:
    - Track model usage
    - Monitor disk space
    - Automatic eviction (LRU)
    - Cache statistics

    Usage:
        cache = CacheManager()

        # Get cache info
        stats = cache.get_stats()
        print(f"Cache size: {stats['total_size_gb']:.1f} GB")

        # Clean old models
        cache.clean(max_age_days=30)

        # Set disk limit
        cache.set_limit(max_size_gb=100)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "unsloth"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"models": {}, "settings": {"max_size_gb": None}}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def track_access(self, model_name: str, model_path: Path):
        """Track model access for LRU eviction."""
        key = str(model_path)

        if key not in self.metadata["models"]:
            # Calculate size
            size_mb = sum(
                f.stat().st_size for f in model_path.rglob("*") if f.is_file()
            ) / 1024**2

            self.metadata["models"][key] = {
                "model_name": model_name,
                "size_mb": size_mb,
                "last_accessed": time.time(),
                "access_count": 1,
                "created": time.time(),
            }
        else:
            self.metadata["models"][key]["last_accessed"] = time.time()
            self.metadata["models"][key]["access_count"] += 1

        self._save_metadata()

        # Check if eviction needed
        self._check_eviction()

    def _check_eviction(self):
        """Check if cache needs eviction."""
        max_size_gb = self.metadata["settings"].get("max_size_gb")
        if not max_size_gb:
            return

        current_size_gb = self.get_cache_size_gb()

        if current_size_gb > max_size_gb:
            self._evict_lru(target_gb=max_size_gb * 0.9)  # Evict to 90%

    def _evict_lru(self, target_gb: float):
        """Evict least recently used models."""
        current_size_gb = self.get_cache_size_gb()

        if current_size_gb <= target_gb:
            return

        # Sort by last access time
        models = sorted(
            self.metadata["models"].items(),
            key=lambda x: x[1]["last_accessed"],
        )

        evicted = []

        for path_str, info in models:
            path = Path(path_str)

            if path.exists():
                shutil.rmtree(path)
                evicted.append(info["model_name"])

            del self.metadata["models"][path_str]

            current_size_gb -= info["size_mb"] / 1024

            if current_size_gb <= target_gb:
                break

        self._save_metadata()

        if evicted:
            print(f"Evicted {len(evicted)} models to free space:")
            for name in evicted:
                print(f"  - {name}")

    def get_cache_size_gb(self) -> float:
        """Get total cache size in GB."""
        return sum(
            info["size_mb"] for info in self.metadata["models"].values()
        ) / 1024

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size_gb = self.get_cache_size_gb()
        num_models = len(self.metadata["models"])

        # Get disk usage
        stat = shutil.disk_usage(self.cache_dir)
        disk_total_gb = stat.total / 1024**3
        disk_free_gb = stat.free / 1024**3
        disk_used_pct = ((stat.total - stat.free) / stat.total) * 100

        return {
            "cache_dir": str(self.cache_dir),
            "num_models": num_models,
            "total_size_gb": total_size_gb,
            "disk_total_gb": disk_total_gb,
            "disk_free_gb": disk_free_gb,
            "disk_used_pct": disk_used_pct,
            "max_size_gb": self.metadata["settings"].get("max_size_gb"),
        }

    def list_models(self) -> List[CachedModel]:
        """List all cached models."""
        models = []

        for path_str, info in self.metadata["models"].items():
            models.append(
                CachedModel(
                    model_name=info["model_name"],
                    path=Path(path_str),
                    size_mb=info["size_mb"],
                    last_accessed=info["last_accessed"],
                    access_count=info["access_count"],
                )
            )

        return sorted(models, key=lambda m: m.last_accessed, reverse=True)

    def clean(
        self,
        max_age_days: Optional[int] = None,
        min_access_count: Optional[int] = None,
    ):
        """Clean cache based on criteria."""
        removed = []

        for path_str, info in list(self.metadata["models"].items()):
            should_remove = False

            if max_age_days:
                age_days = (time.time() - info["last_accessed"]) / 86400
                if age_days > max_age_days:
                    should_remove = True

            if min_access_count:
                if info["access_count"] < min_access_count:
                    should_remove = True

            if should_remove:
                path = Path(path_str)
                if path.exists():
                    shutil.rmtree(path)

                removed.append(info["model_name"])
                del self.metadata["models"][path_str]

        self._save_metadata()

        return removed

    def set_limit(self, max_size_gb: float):
        """Set maximum cache size."""
        self.metadata["settings"]["max_size_gb"] = max_size_gb
        self._save_metadata()
        self._check_eviction()
```

### 2. Fast Model Loader

```python
# unsloth/cache/fast_loader.py
import torch
from pathlib import Path
from typing import Optional, Tuple
import mmap

class FastModelLoader:
    """
    Fast model loading with memory mapping.

    Uses mmap to load models without reading entire file into memory.
    Significantly faster for repeated loads.

    Usage:
        loader = FastModelLoader()
        model = loader.load(model_path)
    """

    def __init__(self):
        self.mmap_cache = {}

    def load(
        self,
        model_path: Path,
        use_mmap: bool = True,
    ) -> torch.nn.Module:
        """
        Load model with optional memory mapping.

        Args:
            model_path: Path to model directory
            use_mmap: Use memory mapping for faster loads

        Returns:
            Loaded model
        """
        safetensors_path = model_path / "model.safetensors"

        if use_mmap and safetensors_path.exists():
            return self._load_mmap(safetensors_path)
        else:
            return self._load_standard(model_path)

    def _load_mmap(self, safetensors_path: Path):
        """Load using memory mapping."""
        from safetensors.torch import load_file

        # Memory map the file
        with open(safetensors_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.mmap_cache[str(safetensors_path)] = mm

        # Load from memory mapped file
        weights = load_file(safetensors_path)

        return weights

    def _load_standard(self, model_path: Path):
        """Standard loading."""
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_path)
```

### 3. Cache CLI

```python
# unsloth/cli/cache.py
import click
from unsloth.cache import CacheManager
from datetime import datetime

@click.group()
def cache():
    """Manage model cache."""
    pass

@cache.command()
def stats():
    """Show cache statistics."""
    manager = CacheManager()
    stats = manager.get_stats()

    click.echo("Cache Statistics")
    click.echo("=" * 50)
    click.echo(f"Cache directory: {stats['cache_dir']}")
    click.echo(f"Models cached:   {stats['num_models']}")
    click.echo(f"Cache size:      {stats['total_size_gb']:.1f} GB")
    click.echo(f"Disk total:      {stats['disk_total_gb']:.1f} GB")
    click.echo(f"Disk free:       {stats['disk_free_gb']:.1f} GB")
    click.echo(f"Disk used:       {stats['disk_used_pct']:.1f}%")

    if stats['max_size_gb']:
        click.echo(f"Cache limit:     {stats['max_size_gb']:.1f} GB")

@cache.command()
def list():
    """List cached models."""
    manager = CacheManager()
    models = manager.list_models()

    if not models:
        click.echo("No models cached")
        return

    click.echo("Cached Models")
    click.echo("=" * 70)

    for model in models:
        last_access = datetime.fromtimestamp(model.last_accessed)
        click.echo(f"{model.model_name}")
        click.echo(f"  Size:         {model.size_mb / 1024:.1f} GB")
        click.echo(f"  Last access:  {last_access.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"  Access count: {model.access_count}")
        click.echo()

@cache.command()
@click.option("--max-age-days", type=int, help="Remove models older than N days")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def clean(max_age_days, yes):
    """Clean old models from cache."""
    manager = CacheManager()

    if max_age_days:
        if not yes:
            click.confirm(
                f"Remove models not accessed in {max_age_days} days?",
                abort=True,
            )

        removed = manager.clean(max_age_days=max_age_days)

        if removed:
            click.echo(f"Removed {len(removed)} models:")
            for name in removed:
                click.echo(f"  - {name}")
        else:
            click.echo("No models removed")

@cache.command()
@click.argument("max_size_gb", type=float)
def set_limit(max_size_gb):
    """Set maximum cache size in GB."""
    manager = CacheManager()
    manager.set_limit(max_size_gb)
    click.echo(f"✓ Cache limit set to {max_size_gb:.1f} GB")

@cache.command()
@click.confirmation_option(prompt="Clear entire cache?")
def clear():
    """Clear entire cache."""
    manager = CacheManager()
    models = manager.list_models()

    for model in models:
        if model.path.exists():
            shutil.rmtree(model.path)

    manager.metadata = {"models": {}, "settings": {}}
    manager._save_metadata()

    click.echo(f"✓ Cleared {len(models)} models from cache")
```

### 4. Automatic Disk Monitoring

```python
# unsloth/cache/monitor.py
import shutil
import warnings
from pathlib import Path

class DiskMonitor:
    """Monitor disk space and warn users."""

    def __init__(self, cache_dir: Path, warning_threshold: float = 0.9):
        self.cache_dir = cache_dir
        self.warning_threshold = warning_threshold

    def check_space(self, required_gb: float) -> bool:
        """
        Check if enough disk space is available.

        Args:
            required_gb: Required space in GB

        Returns:
            True if enough space, False otherwise
        """
        stat = shutil.disk_usage(self.cache_dir)
        free_gb = stat.free / 1024**3

        if free_gb < required_gb:
            warnings.warn(
                f"Insufficient disk space. "
                f"Required: {required_gb:.1f} GB, Available: {free_gb:.1f} GB. "
                f"Run 'unsloth cache clean' to free space."
            )
            return False

        # Warn if getting close to full
        used_pct = ((stat.total - stat.free) / stat.total)
        if used_pct > self.warning_threshold:
            warnings.warn(
                f"Disk {int(used_pct * 100)}% full. "
                f"Consider running 'unsloth cache clean'"
            )

        return True
```

---

## Example Usage

### Cache Management

```python
from unsloth.cache import CacheManager

# Get cache statistics
cache = CacheManager()
stats = cache.get_stats()

print(f"Cache: {stats['total_size_gb']:.1f} GB ({stats['num_models']} models)")
print(f"Disk: {stats['disk_free_gb']:.1f} GB free")

# Set cache limit (auto-evicts old models)
cache.set_limit(max_size_gb=100)

# Clean models not used in 30 days
removed = cache.clean(max_age_days=30)
print(f"Removed {len(removed)} models")

# List cached models
models = cache.list_models()
for model in models[:5]:
    print(f"{model.model_name}: {model.size_mb / 1024:.1f} GB")
```

### CLI Usage

```bash
# Show cache stats
unsloth cache stats

# List cached models
unsloth cache list

# Clean old models
unsloth cache clean --max-age-days 30

# Set cache limit
unsloth cache set-limit 100

# Clear entire cache
unsloth cache clear
```

### Automatic Tracking

```python
from unsloth import FastLanguageModel

# Cache manager automatically tracks usage
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit"
)

# Access is tracked for LRU eviction
# If cache exceeds limit, least recently used models are removed
```

---

## Implementation Plan

### Phase 1: Core Cache Manager (Days 1-4)

**Day 1: Cache Manager**
- Implement CacheManager class
- Add metadata tracking
- Add basic operations

**Day 2: Eviction Policy**
- Implement LRU eviction
- Add disk space checking
- Add limits

**Day 3: Fast Loader**
- Implement memory mapping
- Optimize loading
- Test performance

**Day 4: Testing**
- Test cache operations
- Test eviction
- Test edge cases

### Phase 2: CLI and Integration (Days 5-7)

**Day 5: CLI**
- Implement cache CLI commands
- Add to main CLI
- Test CLI

**Day 6: Integration**
- Integrate with FastLanguageModel
- Add automatic tracking
- Test integration

**Day 7: Disk Monitoring**
- Implement disk monitoring
- Add warnings
- Test monitoring

### Phase 3: Polish (Days 8-10)

**Day 8: Documentation**
- Write cache guide
- Document CLI commands
- Add examples

**Day 9: Testing**
- Test all features
- Test edge cases
- Fix bugs

**Day 10: Final Review**
- Code review
- Performance testing
- Release preparation

### Milestones

| Day | Deliverable |
|-----|-------------|
| 4 | Core cache manager |
| 7 | CLI and integration |
| 10 | Full release |

---

## Backwards Compatibility

### Breaking Changes

None. Cache management is opt-in.

### Migration

Existing caches continue to work. New metadata is created on first use.

---

## Alternatives Considered

### Alternative 1: Manual Cache Management

Leave cache management to users.

**Rejected:**
- Users struggle with disk space
- No standard approach
- Poor experience

### Alternative 2: Use HuggingFace Hub Cache

Rely on HF's cache management.

**Rejected:**
- Limited control
- No Unsloth-specific optimizations
- No fast loading

### Alternative 3: No Cache Limits

Unlimited cache growth.

**Rejected:**
- Can fill disk
- No cleanup mechanism
- Poor user experience

---

## Open Questions

1. **Default cache limit?**
   - 100 GB reasonable?
   - Make it configurable?

2. **Shared cache for quantizations?**
   - Store base model once
   - Apply quantization on load
   - Saves disk space

3. **Cloud cache support?**
   - S3/GCS backend
   - Useful for teams
   - Adds complexity

---

## Success Criteria

- [ ] Cache size tracking works
- [ ] LRU eviction works correctly
- [ ] Disk monitoring warns users
- [ ] CLI commands functional
- [ ] Fast loading shows improvement
- [ ] No disk space issues

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Cache management | Manual | Automated |
| Disk space issues | Common | Rare |
| Model load time | 1.0x | 1.2-1.5x |
| Disk usage | Unlimited | Controlled |

---

## Required Approvals

- [ ] Technical review
- [ ] User testing
- [ ] Documentation review

---

## Rollback Strategy

1. Cache management is opt-in
2. Can disable with environment variable
3. No changes to model format

---

## References

- LRU Cache: https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)
- Memory Mapping: https://docs.python.org/3/library/mmap.html
- Safetensors: https://github.com/huggingface/safetensors

---

*Next: [RFC-0015: Dependency Update Automation](./RFC-0015-dependency-updates.md)*
