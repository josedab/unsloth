# RFC-0008: Multi-GPU Optimization

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 30+ person-days
**Priority:** P2 (Long-term)

---

## Summary

Improve multi-GPU efficiency from 75-85% to 90%+ through custom communication kernels and optimized gradient synchronization.

---

## Motivation

Current multi-GPU scaling:
- 2 GPUs: 1.7-1.8x (85-90% efficiency)
- 4 GPUs: 3.0-3.2x (75-80% efficiency)
- 8 GPUs: 5.5-6.0x (69-75% efficiency)

Standard DDP overhead limits scaling. Custom solutions could improve this significantly.

---

## Detailed Design

### Optimization Strategies

1. **Gradient compression:** Reduce communication volume
2. **Overlapped communication:** Overlap backward with all-reduce
3. **Custom all-reduce:** Optimized for LoRA gradients
4. **Ring all-reduce:** Better topology awareness

### Gradient Compression

```python
class CompressedDDP(nn.Module):
    """DDP with gradient compression for LoRA."""

    def __init__(self, model, compression_ratio=0.1):
        super().__init__()
        self.model = model
        self.compression_ratio = compression_ratio

    def backward_hook(self, grad):
        # Top-K sparsification
        k = int(grad.numel() * self.compression_ratio)
        values, indices = torch.topk(grad.abs().flatten(), k)
        compressed = (values, indices, grad.shape)
        return compressed
```

### Expected Gains

| GPUs | Current | Target | Improvement |
|------|---------|--------|-------------|
| 2 | 85-90% | 95% | +5-10% |
| 4 | 75-80% | 90% | +10-15% |
| 8 | 69-75% | 85% | +10-15% |

---

## Implementation Plan

### Phase 1: Analysis (Days 1-5)
1. Profile current DDP overhead
2. Identify optimization opportunities
3. Design custom communication

### Phase 2: Implementation (Days 6-20)
1. Implement gradient compression
2. Add overlapped communication
3. Benchmark improvements

### Phase 3: Integration (Days 21-30)
1. Integrate with UnslothTrainer
2. Add auto-detection for best strategy
3. Documentation and testing

---

## Risks

- **Complexity:** Multi-GPU is inherently complex
- **Hardware variability:** Different systems behave differently
- **NCCL compatibility:** Custom code must work with NCCL

---

## Success Criteria

- [ ] 90%+ efficiency at 4 GPUs
- [ ] No accuracy degradation
- [ ] Automatic strategy selection
- [ ] Works with existing training scripts

---

## Conclusion

This RFC represents a significant undertaking that would benefit users training on multi-GPU setups. Given the complexity, it should be approached after the foundational improvements (RFC-0001 through RFC-0006) are complete.

---

*Return to [Prioritization Matrix](./00-prioritization-matrix.md)*
