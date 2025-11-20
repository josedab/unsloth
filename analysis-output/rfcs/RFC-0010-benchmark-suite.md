# RFC-0010: Automated Performance Benchmark Suite

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 14 person-days
**Priority:** P1 (Strategic)

---

## Summary

Implement automated performance benchmarking suite to verify Unsloth's 2-5x speedup claims, detect performance regressions in CI, and provide transparent performance comparisons against vanilla transformers and competitors.

---

## Motivation

Unsloth claims 2-5x speedup over standard transformers, but there's no automated verification:

- No benchmark suite to validate claims
- No regression detection in CI
- Cannot track performance over time
- Contributors don't know if changes regress performance
- Users can't reproduce benchmark results

### Current State

Performance claims are based on manual testing:
- No automated benchmarks in repository
- No CI performance gates
- No historical performance tracking
- Claims in README not verifiable

### Impact

| Issue | Effect |
|-------|--------|
| No automated benchmarks | Cannot verify speedup claims |
| No regression detection | Performance can degrade unnoticed |
| No historical data | Cannot track improvements/regressions |
| Manual testing only | Inconsistent, time-consuming |

---

## Detailed Design

### Benchmark Categories

#### 1. Micro-Benchmarks (Component Level)

Test individual optimizations:

```python
# benchmarks/micro/test_attention.py
import torch
from unsloth.models.llama import FastLlamaModel
from transformers.models.llama import LlamaModel
import time

class AttentionBenchmark:
    """Benchmark Flash Attention vs standard attention."""

    def setup(self):
        self.batch_size = 4
        self.seq_len = 2048
        self.hidden_size = 4096
        self.num_heads = 32

        # Create dummy inputs
        self.hidden_states = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_size,
            device="cuda",
            dtype=torch.float16,
        )

    def benchmark_flash_attention(self, iterations=100):
        """Benchmark Unsloth's Flash Attention."""
        model = FastLlamaModel.from_pretrained(
            "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length=2048,
        )

        # Warmup
        for _ in range(10):
            _ = model.model.layers[0].self_attn(self.hidden_states)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            output = model.model.layers[0].self_attn(self.hidden_states)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed / iterations

    def benchmark_standard_attention(self, iterations=100):
        """Benchmark standard transformers attention."""
        model = LlamaModel.from_pretrained(
            "meta-llama/Llama-3-8B",
            torch_dtype=torch.float16,
        )

        # Warmup
        for _ in range(10):
            _ = model.layers[0].self_attn(self.hidden_states)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            output = model.layers[0].self_attn(self.hidden_states)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed / iterations

    def run(self):
        """Run benchmark and compute speedup."""
        unsloth_time = self.benchmark_flash_attention()
        standard_time = self.benchmark_standard_attention()

        speedup = standard_time / unsloth_time

        return {
            "component": "attention",
            "unsloth_ms": unsloth_time * 1000,
            "standard_ms": standard_time * 1000,
            "speedup": speedup,
            "passes": speedup >= 1.5,  # Expect 1.5x+ speedup
        }
```

#### 2. End-to-End Benchmarks (Full Training)

Test complete training workflows:

```python
# benchmarks/e2e/test_qlora_training.py
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from unsloth import UnslothTrainer
from datasets import load_dataset
import time

class QLoRATrainingBenchmark:
    """Benchmark full QLoRA training workflow."""

    def setup(self):
        # Load small dataset
        self.dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

        self.training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            max_steps=100,
            logging_steps=10,
        )

    def benchmark_unsloth(self):
        """Benchmark Unsloth QLoRA training."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            "unsloth/llama-3-8b-bnb-4bit",
            max_seq_length=512,
        )

        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

        trainer = UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=self.dataset,
        )

        start = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - start

        return {
            "total_time": elapsed,
            "steps": trainer.state.global_step,
            "time_per_step": elapsed / trainer.state.global_step,
        }

    def benchmark_standard(self):
        """Benchmark standard transformers + PEFT."""
        from transformers import AutoModelForCausalLM, Trainer
        from peft import get_peft_model, LoraConfig

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3-8B",
            load_in_4bit=True,
        )

        lora_config = LoraConfig(r=16, lora_alpha=16)
        model = get_peft_model(model, lora_config)

        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.dataset,
        )

        start = time.perf_counter()
        trainer.train()
        elapsed = time.perf_counter() - start

        return {
            "total_time": elapsed,
            "steps": trainer.state.global_step,
            "time_per_step": elapsed / trainer.state.global_step,
        }

    def run(self):
        """Run benchmark and compute speedup."""
        unsloth_result = self.benchmark_unsloth()
        standard_result = self.benchmark_standard()

        speedup = standard_result["time_per_step"] / unsloth_result["time_per_step"]

        return {
            "benchmark": "qlora_training",
            "unsloth_time": unsloth_result["total_time"],
            "standard_time": standard_result["total_time"],
            "speedup": speedup,
            "passes": speedup >= 2.0,  # Expect 2x+ speedup
        }
```

#### 3. Memory Benchmarks

Track VRAM usage:

```python
# benchmarks/memory/test_vram_usage.py
import torch
from unsloth import FastLanguageModel

class VRAMBenchmark:
    """Benchmark VRAM usage."""

    def measure_vram(self, model_name, use_unsloth=True):
        """Measure peak VRAM for model."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        if use_unsloth:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name,
                max_seq_length=2048,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Dummy forward pass
        inputs = tokenizer("Test input", return_tensors="pt").to("cuda")
        with torch.no_grad():
            _ = model(**inputs)

        peak_vram = torch.cuda.max_memory_allocated() / 1024**3  # GB

        del model
        torch.cuda.empty_cache()

        return peak_vram

    def run(self):
        """Compare VRAM usage."""
        unsloth_vram = self.measure_vram("unsloth/llama-3-8b-bnb-4bit", use_unsloth=True)
        standard_vram = self.measure_vram("meta-llama/Llama-3-8B", use_unsloth=False)

        reduction = (standard_vram - unsloth_vram) / standard_vram * 100

        return {
            "benchmark": "vram_usage",
            "unsloth_gb": unsloth_vram,
            "standard_gb": standard_vram,
            "reduction_pct": reduction,
            "passes": reduction >= 60,  # Expect 60%+ reduction
        }
```

### CI Integration

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    labels: [run-benchmarks]

jobs:
  benchmarks:
    if: github.ref == 'refs/heads/main' || contains(github.event.pull_request.labels.*.name, 'run-benchmarks')
    runs-on: [self-hosted, gpu, benchmark]

    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: pip install -e ".[dev,benchmark]"

      - name: Run micro-benchmarks
        run: |
          pytest benchmarks/micro -v \
            --benchmark-json=results-micro.json

      - name: Run E2E benchmarks
        run: |
          pytest benchmarks/e2e -v \
            --benchmark-json=results-e2e.json

      - name: Check for regressions
        run: |
          python scripts/check_regressions.py \
            --current=results-e2e.json \
            --baseline=benchmarks/baselines/main.json \
            --threshold=0.95  # Fail if <95% of baseline

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results-*.json

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./results-e2e.json');
            const comment = `## Benchmark Results\n\n${formatResults(results)}`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

### Regression Detection

```python
# scripts/check_regressions.py
import json
import sys

def check_regressions(current_file, baseline_file, threshold=0.95):
    """Check if current performance meets baseline threshold."""

    with open(current_file) as f:
        current = json.load(f)

    with open(baseline_file) as f:
        baseline = json.load(f)

    regressions = []

    for benchmark_name, current_result in current.items():
        if benchmark_name not in baseline:
            continue

        baseline_result = baseline[benchmark_name]

        # Check speedup
        speedup_ratio = current_result["speedup"] / baseline_result["speedup"]

        if speedup_ratio < threshold:
            regressions.append({
                "benchmark": benchmark_name,
                "current": current_result["speedup"],
                "baseline": baseline_result["speedup"],
                "ratio": speedup_ratio,
            })

    if regressions:
        print("⚠️  Performance regressions detected:")
        for reg in regressions:
            print(f"  - {reg['benchmark']}: {reg['current']:.2f}x (was {reg['baseline']:.2f}x)")
        sys.exit(1)
    else:
        print("✅ No performance regressions detected")
        sys.exit(0)
```

---

## Example Usage

### Running Benchmarks Locally

```bash
# Run all benchmarks
pytest benchmarks/ -v

# Run specific category
pytest benchmarks/micro -v

# Run with detailed output
pytest benchmarks/e2e -v --benchmark-verbose

# Compare with baseline
python scripts/check_regressions.py \
  --current=results-e2e.json \
  --baseline=benchmarks/baselines/main.json
```

### Viewing Historical Results

```bash
# Generate performance report
python scripts/generate_report.py \
  --results=benchmarks/history/ \
  --output=performance_report.html
```

---

## Implementation Plan

### Phase 1: Infrastructure (Days 1-3)

**Day 1: Setup**
- Create `benchmarks/` directory structure
- Add pytest-benchmark dependency
- Configure CI workflow

**Day 2: Utilities**
- Write regression detection script
- Create baseline storage system
- Add result formatting utilities

**Day 3: CI Integration**
- Set up GPU runner for benchmarks
- Configure benchmark job
- Test workflow

### Phase 2: Micro-Benchmarks (Days 4-7)

**Day 4: Attention**
- Flash Attention vs standard
- Different sequence lengths
- Memory usage

**Day 5: LoRA Operations**
- Forward pass benchmark
- Backward pass benchmark
- Gradient computation

**Day 6: Quantization**
- 4-bit vs 8-bit vs 16-bit
- Dequantization speed
- Memory comparison

**Day 7: Kernels**
- Cross-entropy loss
- RMS norm
- RoPE embeddings

### Phase 3: E2E Benchmarks (Days 8-11)

**Day 8: Training**
- QLoRA full training
- Standard LoRA training
- DPO training

**Day 9: Inference**
- Generation speed
- Batch inference
- Streaming

**Day 10: Saving/Loading**
- Save to GGUF
- Merge and save
- Load time

**Day 11: Multi-GPU**
- DDP efficiency
- Gradient sync overhead

### Phase 4: Reporting and Tracking (Days 12-14)

**Day 12: Visualization**
- HTML report generation
- Historical charts
- Comparison tables

**Day 13: Automation**
- Baseline update workflow
- Regression alerting
- Slack/Discord notifications

**Day 14: Documentation**
- Benchmark guide
- Adding new benchmarks
- Interpreting results

### Milestones

| Day | Deliverable |
|-----|-------------|
| 3 | CI infrastructure ready |
| 7 | Micro-benchmarks complete |
| 11 | E2E benchmarks complete |
| 14 | Full automation live |

---

## Backwards Compatibility

### Breaking Changes

None. Benchmarks are isolated from production code.

### Requirements

New dependencies for benchmarking:
```toml
[project.optional-dependencies]
benchmark = [
    "pytest-benchmark>=4.0.0",
    "pandas>=1.5.0",
    "matplotlib>=3.6.0",
]
```

---

## Alternatives Considered

### Alternative 1: Manual Benchmarking

Continue with manual performance testing.

**Rejected:**
- Not scalable
- No regression detection
- Inconsistent methodology

### Alternative 2: Third-Party Benchmark Service

Use external benchmarking service (e.g., CodSpeed).

**Rejected:**
- Cost concerns
- Less control
- May not support GPU benchmarks

### Alternative 3: Benchmark on Every PR

Run full benchmark suite on all PRs.

**Rejected:**
- Too expensive (GPU time)
- Too slow (blocks PRs)
- Label-based opt-in better

---

## Open Questions

1. **Which GPU for baseline?**
   - T4 (widely available)
   - A100 (high performance)
   - Both?

2. **How often to update baselines?**
   - Every release
   - Monthly
   - On major changes

3. **Should benchmarks block PRs?**
   - Fail on regression
   - Warning only
   - Manual review

4. **Track competitors?**
   - Benchmark against FastChat, vLLM
   - Adds complexity
   - Useful for marketing

---

## Success Criteria

- [ ] Micro-benchmarks for all optimized components
- [ ] E2E benchmarks for common workflows
- [ ] CI runs benchmarks on main and labeled PRs
- [ ] Regression detection prevents performance degradation
- [ ] Historical tracking shows improvements over time
- [ ] Public benchmark results validate speedup claims

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Automated benchmarks | 0 | 20+ |
| Performance verification | Manual | Automated |
| Regression detection | None | CI-enforced |
| Speedup validation | Anecdotal | Data-driven |

---

## Required Approvals

- [ ] GPU runner access for benchmarks
- [ ] CI budget approval
- [ ] Baseline methodology approval

---

## Rollback Strategy

1. Disable benchmark job in GitHub Actions
2. Remove from required checks
3. Benchmarks remain but don't block

---

## References

- pytest-benchmark: https://pytest-benchmark.readthedocs.io/
- MLPerf Benchmarks: https://mlcommons.org/en/training-normal-11/
- HuggingFace Benchmarks: https://huggingface.co/docs/transformers/benchmarks
- CodSpeed: https://codspeed.io/

---

*Next: [RFC-0011: Memory Leak Detection and VRAM Profiling](./RFC-0011-memory-profiling.md)*
