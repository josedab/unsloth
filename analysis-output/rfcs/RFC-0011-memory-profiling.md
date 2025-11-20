# RFC-0011: Memory Leak Detection and VRAM Profiling

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 10 person-days
**Priority:** P1 (Strategic)

---

## Summary

Implement automated memory leak detection and VRAM profiling tools to identify memory leaks, optimize VRAM usage, and provide users with detailed memory insights for debugging OOM (Out-of-Memory) errors.

---

## Motivation

Memory management is critical for LLM training, but Unsloth lacks tooling to detect and diagnose memory issues:

- Users frequently hit OOM errors without understanding why
- No profiling tools to identify VRAM bottlenecks
- Potential memory leaks in model loading/saving not detected
- No guidance on optimal batch size for available VRAM
- Manual memory tracking with `torch.cuda.memory_allocated()` insufficient

### Current Issues

```python
# From GitHub issues - common pattern
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 15.90 GiB total capacity; 13.45 GiB already allocated;
1.23 GiB free; 13.67 GiB reserved in total by PyTorch)
```

Users don't know:
- What's using the 13.45 GiB
- Whether there's a leak
- Optimal batch size for their GPU
- Which model components use most VRAM

### Impact

| Issue | Effect |
|-------|--------|
| No leak detection | Memory grows unbounded over time |
| No profiling tools | Cannot optimize VRAM usage |
| Poor error messages | Users can't debug OOM errors |
| No VRAM recommendations | Suboptimal configurations |

---

## Detailed Design

### 1. Memory Leak Detector

Automatic leak detection in training loops:

```python
# unsloth/memory/leak_detector.py
import torch
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
import weakref

@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    step: int
    allocated_mb: float
    reserved_mb: float
    num_tensors: int
    largest_tensors: List[tuple]  # (size_mb, shape, dtype)

class MemoryLeakDetector:
    """
    Detect memory leaks during training.

    Usage:
        detector = MemoryLeakDetector()

        for step in range(num_steps):
            # Training code
            loss.backward()
            optimizer.step()

            # Check for leaks
            if step % 100 == 0:
                detector.snapshot(step)
                if detector.has_leak():
                    detector.report()
    """

    def __init__(
        self,
        leak_threshold_mb: float = 100.0,
        window_size: int = 10,
    ):
        self.leak_threshold_mb = leak_threshold_mb
        self.window_size = window_size
        self.snapshots: List[MemorySnapshot] = []
        self.tensor_tracker = TensorTracker()

    def snapshot(self, step: int) -> MemorySnapshot:
        """Take memory snapshot at current step."""
        gc.collect()
        torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2

        # Track tensors
        tensors = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                if size_mb > 1.0:  # Only track tensors > 1MB
                    tensors.append((size_mb, obj.shape, obj.dtype))

        tensors.sort(reverse=True)

        snapshot = MemorySnapshot(
            step=step,
            allocated_mb=allocated,
            reserved_mb=reserved,
            num_tensors=len(tensors),
            largest_tensors=tensors[:10],
        )

        self.snapshots.append(snapshot)

        # Keep only recent snapshots
        if len(self.snapshots) > self.window_size * 2:
            self.snapshots = self.snapshots[-self.window_size * 2:]

        return snapshot

    def has_leak(self) -> bool:
        """Check if memory is leaking."""
        if len(self.snapshots) < self.window_size:
            return False

        # Compare recent window to earlier window
        recent = self.snapshots[-self.window_size:]
        earlier = self.snapshots[-self.window_size * 2:-self.window_size]

        recent_avg = sum(s.allocated_mb for s in recent) / len(recent)
        earlier_avg = sum(s.allocated_mb for s in earlier) / len(earlier)

        growth = recent_avg - earlier_avg

        return growth > self.leak_threshold_mb

    def report(self) -> str:
        """Generate leak report."""
        if not self.has_leak():
            return "No memory leak detected."

        recent = self.snapshots[-self.window_size:]
        earlier = self.snapshots[-self.window_size * 2:-self.window_size]

        recent_avg = sum(s.allocated_mb for s in recent) / len(recent)
        earlier_avg = sum(s.allocated_mb for s in earlier) / len(earlier)
        growth = recent_avg - earlier_avg

        report = [
            "⚠️  MEMORY LEAK DETECTED",
            f"",
            f"Average memory growth: {growth:.1f} MB",
            f"Recent average: {recent_avg:.1f} MB",
            f"Earlier average: {earlier_avg:.1f} MB",
            f"",
            f"Largest tensors in recent snapshot:",
        ]

        latest = self.snapshots[-1]
        for i, (size_mb, shape, dtype) in enumerate(latest.largest_tensors[:5], 1):
            report.append(f"  {i}. {size_mb:.1f} MB - shape {shape} - {dtype}")

        report.append(f"")
        report.append(f"Recommendations:")
        report.append(f"  - Check for tensors not being released")
        report.append(f"  - Verify gradient accumulation is clearing properly")
        report.append(f"  - Look for caching that grows unbounded")

        return "\n".join(report)

class TensorTracker:
    """Track tensor allocations to find leaks."""

    def __init__(self):
        self.tensors = weakref.WeakSet()

    def track(self, tensor: torch.Tensor):
        """Track a tensor allocation."""
        self.tensors.add(tensor)

    def get_alive_tensors(self) -> List[torch.Tensor]:
        """Get all tracked tensors still alive."""
        return list(self.tensors)
```

### 2. VRAM Profiler

Detailed VRAM usage breakdown:

```python
# unsloth/memory/profiler.py
import torch
from contextlib import contextmanager
from typing import Dict, Optional
import time

class VRAMProfiler:
    """
    Profile VRAM usage during model operations.

    Usage:
        profiler = VRAMProfiler()

        with profiler.profile("model_loading"):
            model, tokenizer = FastLanguageModel.from_pretrained(...)

        with profiler.profile("forward_pass"):
            outputs = model(**inputs)

        profiler.print_summary()
    """

    def __init__(self):
        self.measurements: Dict[str, Dict] = {}
        torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def profile(self, name: str):
        """Profile a code block."""
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        start_time = time.perf_counter()

        torch.cuda.reset_peak_memory_stats()

        try:
            yield
        finally:
            torch.cuda.synchronize()

            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            elapsed = time.perf_counter() - start_time

            self.measurements[name] = {
                "allocated_mb": (end_allocated - start_allocated) / 1024**2,
                "reserved_mb": (end_reserved - start_reserved) / 1024**2,
                "peak_allocated_mb": peak_allocated / 1024**2,
                "peak_reserved_mb": peak_reserved / 1024**2,
                "time_sec": elapsed,
            }

    def print_summary(self):
        """Print profiling summary."""
        print("\n" + "="*60)
        print("VRAM Profiling Summary")
        print("="*60)

        total_allocated = 0
        total_time = 0

        for name, metrics in self.measurements.items():
            print(f"\n{name}:")
            print(f"  Allocated: {metrics['allocated_mb']:.1f} MB")
            print(f"  Peak:      {metrics['peak_allocated_mb']:.1f} MB")
            print(f"  Time:      {metrics['time_sec']:.2f} sec")

            total_allocated += metrics['allocated_mb']
            total_time += metrics['time_sec']

        print(f"\n{'='*60}")
        print(f"Total Allocated: {total_allocated:.1f} MB")
        print(f"Total Time:      {total_time:.2f} sec")
        print(f"{'='*60}\n")

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on profile."""
        recommendations = []

        total_allocated = torch.cuda.memory_allocated() / 1024**2
        total_capacity = torch.cuda.get_device_properties(0).total_memory / 1024**2
        usage_pct = (total_allocated / total_capacity) * 100

        if usage_pct > 90:
            recommendations.append(
                "⚠️  VRAM usage >90% - reduce batch size or sequence length"
            )

        if usage_pct > 80:
            recommendations.append(
                "Consider enabling gradient checkpointing for memory savings"
            )

        # Check for fragmentation
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        fragmentation = ((reserved - allocated) / reserved) * 100 if reserved > 0 else 0

        if fragmentation > 20:
            recommendations.append(
                f"Memory fragmentation detected ({fragmentation:.1f}%) - "
                f"consider calling torch.cuda.empty_cache()"
            )

        return recommendations
```

### 3. Automatic Batch Size Finder

Find optimal batch size for available VRAM:

```python
# unsloth/memory/batch_size_finder.py
import torch
from typing import Callable, Optional

class BatchSizeFinder:
    """
    Automatically find optimal batch size for available VRAM.

    Usage:
        finder = BatchSizeFinder()

        optimal_bs = finder.find_batch_size(
            train_fn=lambda bs: trainer.train(batch_size=bs),
            start_batch_size=16,
        )

        print(f"Optimal batch size: {optimal_bs}")
    """

    def __init__(self, safety_margin: float = 0.9):
        """
        Args:
            safety_margin: Use this fraction of max batch size (default 0.9 = 90%)
        """
        self.safety_margin = safety_margin

    def find_batch_size(
        self,
        train_fn: Callable[[int], None],
        start_batch_size: int = 1,
        max_batch_size: int = 512,
    ) -> int:
        """
        Binary search for maximum batch size that fits in VRAM.

        Args:
            train_fn: Function that runs one training step with given batch size
            start_batch_size: Starting batch size
            max_batch_size: Maximum batch size to try

        Returns:
            Optimal batch size
        """
        low = start_batch_size
        high = max_batch_size
        best = start_batch_size

        print(f"Finding optimal batch size (range: {low}-{high})...")

        while low <= high:
            mid = (low + high) // 2

            print(f"  Trying batch size {mid}...", end=" ")

            try:
                torch.cuda.empty_cache()
                train_fn(mid)
                torch.cuda.synchronize()

                print("✓ Success")
                best = mid
                low = mid + 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("✗ OOM")
                    high = mid - 1
                else:
                    raise

        optimal = int(best * self.safety_margin)
        print(f"\nOptimal batch size: {optimal} (with {self.safety_margin*100:.0f}% safety margin)")

        return optimal
```

### 4. Memory Dashboard

Real-time memory monitoring:

```python
# unsloth/memory/dashboard.py
from typing import Optional
import torch

class MemoryDashboard:
    """Display real-time memory usage."""

    @staticmethod
    def print_status(step: Optional[int] = None):
        """Print current memory status."""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        usage_pct = (allocated / total) * 100
        reserved_pct = (reserved / total) * 100

        # Status indicator
        if usage_pct < 70:
            status = "✓"
        elif usage_pct < 90:
            status = "⚠"
        else:
            status = "✗"

        step_str = f"Step {step:5d} | " if step is not None else ""

        print(
            f"{status} {step_str}"
            f"VRAM: {allocated:.2f} GB / {total:.2f} GB ({usage_pct:.1f}%) | "
            f"Reserved: {reserved:.2f} GB ({reserved_pct:.1f}%)"
        )
```

---

## Example Usage

### Memory Leak Detection

```python
from unsloth import FastLanguageModel
from unsloth.memory import MemoryLeakDetector

model, tokenizer = FastLanguageModel.from_pretrained(...)

detector = MemoryLeakDetector(leak_threshold_mb=100)

for step in range(1000):
    # Training
    loss = trainer.training_step(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Check for leaks every 100 steps
    if step % 100 == 0:
        detector.snapshot(step)
        if detector.has_leak():
            print(detector.report())
            break
```

### VRAM Profiling

```python
from unsloth.memory import VRAMProfiler

profiler = VRAMProfiler()

with profiler.profile("model_loading"):
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/llama-3-8b-bnb-4bit"
    )

with profiler.profile("lora_application"):
    model = FastLanguageModel.get_peft_model(model, r=16)

with profiler.profile("forward_pass"):
    outputs = model(**inputs)

profiler.print_summary()
recommendations = profiler.get_recommendations()
for rec in recommendations:
    print(rec)
```

### Automatic Batch Size Finding

```python
from unsloth.memory import BatchSizeFinder

finder = BatchSizeFinder(safety_margin=0.9)

def train_step(batch_size):
    # Dummy training step
    trainer.train(batch_size=batch_size, max_steps=5)

optimal_bs = finder.find_batch_size(
    train_fn=train_step,
    start_batch_size=1,
    max_batch_size=64,
)

print(f"Use batch_size={optimal_bs} for training")
```

---

## Implementation Plan

### Phase 1: Core Tools (Days 1-4)

**Day 1: Memory Leak Detector**
- Implement MemorySnapshot
- Implement MemoryLeakDetector
- Add tensor tracking

**Day 2: VRAM Profiler**
- Implement VRAMProfiler
- Add context manager
- Add recommendations

**Day 3: Batch Size Finder**
- Implement binary search
- Add safety margin
- Test with different models

**Day 4: Dashboard**
- Implement real-time display
- Add progress integration
- Format output

### Phase 2: Integration (Days 5-7)

**Day 5: Trainer Integration**
- Add leak detection to UnslothTrainer
- Add automatic profiling option
- Add batch size finder CLI

**Day 6: CLI Tools**
- `unsloth-profile` command
- `unsloth-find-batch-size` command
- `unsloth-memory-test` command

**Day 7: Testing**
- Test leak detection
- Test profiler accuracy
- Test batch size finder

### Phase 3: Documentation (Days 8-10)

**Day 8: User Guide**
- Memory optimization guide
- Troubleshooting OOM errors
- Profiling best practices

**Day 9: Examples**
- Example notebooks
- Common patterns
- Advanced usage

**Day 10: Final Review**
- Code review
- Documentation review
- Release preparation

### Milestones

| Day | Deliverable |
|-----|-------------|
| 4 | Core tools complete |
| 7 | Trainer integration |
| 10 | Full release ready |

---

## Backwards Compatibility

### Breaking Changes

None. All tools are opt-in.

### Usage

Tools can be enabled via environment variables:
```bash
# Enable automatic leak detection
export UNSLOTH_DETECT_LEAKS=1

# Enable profiling
export UNSLOTH_PROFILE_MEMORY=1
```

---

## Alternatives Considered

### Alternative 1: Use PyTorch Profiler Only

Rely on `torch.profiler` for memory profiling.

**Rejected:**
- Too complex for typical users
- Doesn't detect leaks automatically
- No Unsloth-specific guidance

### Alternative 2: Manual Tracking

Leave memory tracking to users.

**Rejected:**
- Users struggle with OOM errors
- No standardized approach
- Poor user experience

### Alternative 3: Third-Party Tools

Use tools like `memory_profiler`.

**Rejected:**
- Not GPU-specific
- Not integrated with training
- Additional dependency

---

## Open Questions

1. **Should leak detection be enabled by default?**
   - Pro: Catches issues early
   - Con: Small overhead

2. **How to handle multi-GPU?**
   - Track each GPU separately
   - Aggregate view
   - Both?

3. **Store profiling history?**
   - Save to disk
   - Upload to cloud
   - In-memory only

---

## Success Criteria

- [ ] Leak detector catches growing memory
- [ ] Profiler shows accurate VRAM breakdown
- [ ] Batch size finder works on T4, A100, H100
- [ ] OOM errors reduced by 50%+ via better guidance
- [ ] Documentation covers common memory issues

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Memory debugging tools | 0 | 4 |
| OOM error issues | High | 50% reduction |
| Batch size optimization | Manual | Automated |
| Leak detection | Manual | Automated |

---

## Required Approvals

- [ ] Technical review
- [ ] Documentation review
- [ ] User testing

---

## Rollback Strategy

1. Tools are opt-in, can be disabled
2. No changes to core training code
3. Can be removed without breaking changes

---

## References

- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- CUDA Memory Profiling: https://developer.nvidia.com/blog/profiling-cuda-applications/
- Memory Leak Detection: https://github.com/pytorch/pytorch/issues/13246

---

*Next: [RFC-0012: Centralized Configuration Management](./RFC-0012-config-management.md)*
