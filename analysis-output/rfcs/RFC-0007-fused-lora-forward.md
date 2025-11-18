# RFC-0007: Fused LoRA Forward Pass

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 20 person-days
**Priority:** P2 (Long-term)

---

## Summary

Implement a fused Triton kernel for LoRA forward pass to match the optimization level of the backward pass, achieving 1.3-1.5x forward speedup.

---

## Motivation

The LoRA backward pass is optimized with a 5-10x speedup, but the forward pass uses standard PyTorch operations:

```python
# Current forward (not optimized)
Y = matmul(X, W)                    # Base
lora = matmul(matmul(X, A), B)      # LoRA (2 matmuls)
Y = Y + scaling * lora              # Combine
```

This leaves performance on the table.

---

## Detailed Design

### Fused Forward Kernel

```python
@triton.jit
def fused_lora_forward_kernel(
    X_ptr, W_ptr, A_ptr, B_ptr, Y_ptr,
    M, N, K, R,  # Dimensions
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y = X @ W + scaling * X @ A @ B

    Fuses:
    1. Base matmul: X @ W
    2. LoRA matmul 1: X @ A
    3. LoRA matmul 2: (X @ A) @ B
    4. Scaling and addition

    Into single kernel with shared X loading.
    """
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load X tile (shared between base and LoRA)
    # ... implementation details
```

### Expected Gains

| Operation | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Forward pass | 1.0x | 1.3-1.5x | 30-50% |

### Approach

1. **Single X load:** Load input once, use for both paths
2. **Fused accumulation:** Add base and LoRA in registers
3. **Pipelined loads:** Overlap memory and compute

---

## Implementation Plan

### Phase 1: Kernel Development (Days 1-8)
1. Implement basic fused kernel
2. Handle edge cases (batch size, dimensions)
3. Add autotuning

### Phase 2: Integration (Days 9-14)
1. Integrate with LoRA_MLP class
2. Add backward compatibility
3. Benchmark against current

### Phase 3: Optimization (Days 15-20)
1. Profile and tune
2. Add to all LoRA modules
3. Documentation

---

## Risks

- **Kernel complexity:** LoRA matmul shapes vary
- **Memory constraints:** May need more shared memory
- **Numerical precision:** Must match PyTorch exactly

---

## Success Criteria

- [ ] 1.3x minimum forward speedup
- [ ] Numerical equivalence to PyTorch
- [ ] No memory regression
- [ ] All existing tests pass

---

*Next: [RFC-0008: Multi-GPU Optimization](./RFC-0008-multi-gpu.md)*
