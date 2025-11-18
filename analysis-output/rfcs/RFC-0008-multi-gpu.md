# RFC-0008: Multi-GPU Optimization

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 30+ person-days
**Priority:** P2 (Long-term)

---

## Summary

Improve multi-GPU training efficiency from 75-85% to 90%+ through custom gradient communication, optimized synchronization, and LoRA-aware distributed strategies.

---

## Motivation

Unsloth currently uses standard PyTorch DDP for multi-GPU training, which has significant overhead:

| GPUs | Theoretical Speedup | Actual Speedup | Efficiency |
|------|---------------------|----------------|------------|
| 1 | 1.0x | 1.0x | 100% |
| 2 | 2.0x | 1.7-1.8x | 85-90% |
| 4 | 4.0x | 3.0-3.2x | 75-80% |
| 8 | 8.0x | 5.5-6.0x | 69-75% |

The 10-25% efficiency loss at 4+ GPUs comes from:
- Gradient synchronization overhead (all-reduce)
- Communication blocking compute
- Inefficient bandwidth utilization

### Impact

For users training on multi-GPU setups:
- 4 GPU cluster: 20-25% compute wasted
- 8 GPU cluster: 25-31% compute wasted
- Cloud costs significantly higher than necessary

### Why Standard DDP Is Suboptimal

1. **Full gradient all-reduce:** All gradients synchronized, even when sparse
2. **Blocking communication:** Compute waits for communication
3. **No LoRA awareness:** Treats LoRA gradients same as full gradients
4. **Generic topology:** No optimization for specific network topology

---

## Detailed Design

### Strategy 1: Gradient Compression

LoRA gradients are often sparse - only top values matter:

```python
class CompressedAllReduce:
    """
    Compress gradients before all-reduce.

    Only synchronize top-K gradient values.
    """

    def __init__(self, compression_ratio=0.1):
        self.compression_ratio = compression_ratio
        self.error_feedback = {}  # For error feedback correction

    def compress(self, tensor, name):
        """Compress gradient using top-K sparsification."""
        numel = tensor.numel()
        k = int(numel * self.compression_ratio)

        # Get top-K values and indices
        values, indices = torch.topk(tensor.abs().flatten(), k)
        signs = torch.sign(tensor.flatten()[indices])

        # Error feedback: add previous compression error
        if name in self.error_feedback:
            tensor = tensor + self.error_feedback[name]

        # Store compression error for next iteration
        compressed = torch.zeros_like(tensor.flatten())
        compressed[indices] = tensor.flatten()[indices]
        self.error_feedback[name] = tensor.flatten() - compressed

        return values * signs, indices

    def decompress(self, values, indices, shape):
        """Decompress gradient back to full tensor."""
        tensor = torch.zeros(shape.numel(), dtype=values.dtype, device=values.device)
        tensor[indices] = values
        return tensor.view(shape)
```

### Strategy 2: Overlapped Communication

Overlap backward computation with gradient communication:

```python
class OverlappedDDP(nn.Module):
    """
    DDP with overlapped communication and computation.
    """

    def __init__(self, model, bucket_size_mb=25):
        super().__init__()
        self.model = model
        self.bucket_size = bucket_size_mb * 1024 * 1024

        # Create gradient buckets
        self.buckets = self._create_buckets()

        # Communication stream
        self.comm_stream = torch.cuda.Stream()

        # Register hooks
        self._register_hooks()

    def _create_buckets(self):
        """Group parameters into communication buckets."""
        buckets = []
        current_bucket = []
        current_size = 0

        for param in self.model.parameters():
            if param.requires_grad:
                param_size = param.numel() * param.element_size()
                if current_size + param_size > self.bucket_size:
                    buckets.append(current_bucket)
                    current_bucket = [param]
                    current_size = param_size
                else:
                    current_bucket.append(param)
                    current_size += param_size

        if current_bucket:
            buckets.append(current_bucket)

        return buckets

    def _register_hooks(self):
        """Register backward hooks for overlapped communication."""
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_hook(
                    lambda grad, idx=bucket_idx: self._bucket_hook(grad, idx)
                )

    def _bucket_hook(self, grad, bucket_idx):
        """Start all-reduce when bucket is ready."""
        bucket = self.buckets[bucket_idx]

        # Check if all gradients in bucket are ready
        if all(p.grad is not None for p in bucket):
            # Launch all-reduce on communication stream
            with torch.cuda.stream(self.comm_stream):
                grads = [p.grad for p in bucket]
                flat_grad = torch.cat([g.flatten() for g in grads])

                dist.all_reduce(flat_grad)
                flat_grad /= dist.get_world_size()

                # Copy back
                offset = 0
                for p in bucket:
                    numel = p.grad.numel()
                    p.grad.copy_(flat_grad[offset:offset+numel].view(p.grad.shape))
                    offset += numel

        return grad
```

### Strategy 3: LoRA-Aware Distribution

Optimize for LoRA's unique gradient structure:

```python
class LoRAAwareDDP:
    """
    DDP optimized for LoRA training.

    Key insight: LoRA gradients (A, B matrices) are much smaller
    than base model gradients and can be synchronized faster.
    """

    def __init__(self, model):
        self.model = model
        self.lora_params = []
        self.base_params = []

        # Separate LoRA and base parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    self.lora_params.append(param)
                else:
                    self.base_params.append(param)

    def sync_gradients(self):
        """
        Synchronize gradients with different strategies.

        LoRA params: All-reduce immediately (small)
        Base params: Skip (frozen in LoRA training)
        """
        # LoRA gradients: full precision, immediate sync
        if self.lora_params:
            lora_grads = [p.grad for p in self.lora_params if p.grad is not None]
            if lora_grads:
                flat = torch.cat([g.flatten() for g in lora_grads])
                dist.all_reduce(flat)
                flat /= dist.get_world_size()

                # Copy back
                offset = 0
                for g in lora_grads:
                    numel = g.numel()
                    g.copy_(flat[offset:offset+numel].view(g.shape))
                    offset += numel
```

### Strategy 4: Ring All-Reduce Optimization

Custom ring all-reduce for better bandwidth utilization:

```python
def ring_allreduce(tensor, group=None):
    """
    Ring all-reduce with better bandwidth utilization.

    Instead of tree-based reduction, use ring topology
    for better scaling on high-bandwidth interconnects.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Split tensor into chunks
    chunks = tensor.chunk(world_size)

    # Scatter-reduce phase
    for i in range(world_size - 1):
        send_idx = (rank - i) % world_size
        recv_idx = (rank - i - 1) % world_size

        send_chunk = chunks[send_idx]
        recv_chunk = torch.empty_like(chunks[recv_idx])

        # Send and receive
        send_op = dist.isend(send_chunk, (rank + 1) % world_size, group)
        dist.recv(recv_chunk, (rank - 1) % world_size, group)
        send_op.wait()

        # Reduce
        chunks[recv_idx] += recv_chunk

    # All-gather phase
    for i in range(world_size - 1):
        send_idx = (rank - i + 1) % world_size
        recv_idx = (rank - i) % world_size

        send_chunk = chunks[send_idx]

        # Send and receive
        send_op = dist.isend(send_chunk, (rank + 1) % world_size, group)
        dist.recv(chunks[recv_idx], (rank - 1) % world_size, group)
        send_op.wait()

    # Reconstruct tensor
    return torch.cat(chunks)
```

### Integration

```python
# unsloth/distributed.py

class UnslothDDP:
    """
    Optimized DDP for Unsloth training.

    Combines multiple strategies:
    1. LoRA-aware gradient handling
    2. Overlapped communication
    3. Optional gradient compression
    """

    def __init__(
        self,
        model,
        compression_ratio=None,  # None = no compression
        overlap=True,
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        self.overlap = overlap

        # Initialize strategies
        if compression_ratio:
            self.compressor = CompressedAllReduce(compression_ratio)

        if overlap:
            self.ddp = OverlappedDDP(model)
        else:
            self.ddp = LoRAAwareDDP(model)

    @staticmethod
    def from_model(model, **kwargs):
        """Create optimized DDP from model."""
        return UnslothDDP(model, **kwargs)
```

---

## Example Usage

```python
from unsloth import FastLanguageModel
from unsloth.distributed import UnslothDDP

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Apply LoRA
model = FastLanguageModel.get_peft_model(model, r=16)

# Wrap with optimized DDP
model = UnslothDDP.from_model(
    model,
    compression_ratio=0.1,  # 10x compression
    overlap=True,           # Overlapped communication
)

# Training proceeds normally
trainer = UnslothTrainer(model=model, ...)
trainer.train()
```

### Automatic Strategy Selection

```python
# Auto-select best strategy based on setup
model = UnslothDDP.auto(
    model,
    world_size=4,
    interconnect="nvlink",  # or "ethernet", "infiniband"
)
```

---

## Implementation Plan

### Phase 1: Analysis and Design (Days 1-5)

**Days 1-2: Profiling**
- Profile current DDP overhead
- Identify communication bottlenecks
- Measure gradient statistics

**Days 3-5: Design**
- Design UnslothDDP interface
- Plan integration with trainer
- Define strategy selection logic

### Phase 2: Core Implementation (Days 6-15)

**Days 6-8: Overlapped Communication**
- Implement bucket-based all-reduce
- Add backward hooks
- Test correctness

**Days 9-11: Gradient Compression**
- Implement top-K sparsification
- Add error feedback
- Test accuracy

**Days 12-15: LoRA Optimization**
- Implement LoRA-aware sync
- Optimize for small gradients
- Test with various ranks

### Phase 3: Optimization (Days 16-22)

**Days 16-18: Performance Tuning**
- Tune bucket sizes
- Optimize compression ratio
- Profile and eliminate bottlenecks

**Days 19-22: Advanced Features**
- Ring all-reduce option
- Topology awareness
- Auto-selection logic

### Phase 4: Testing and Integration (Days 23-30)

**Days 23-25: Testing**
- Multi-GPU correctness tests
- Convergence validation
- Stress testing

**Days 26-28: Integration**
- Integrate with UnslothTrainer
- Add documentation
- Create examples

**Days 29-30: Review**
- Code review
- Final benchmarking
- Merge preparation

### Milestones

| Day | Deliverable |
|-----|-------------|
| 5 | Design complete |
| 15 | Core implementation |
| 22 | Optimization complete |
| 30 | Ready for merge |

---

## Backwards Compatibility

### Breaking Changes

None. Opt-in optimization.

### Fallback

```python
# Use standard DDP if issues
if os.environ.get("UNSLOTH_USE_STANDARD_DDP"):
    model = DDP(model)
else:
    model = UnslothDDP(model)
```

### Convergence Guarantee

Must match standard DDP convergence:

```python
def test_convergence():
    # Train with standard DDP
    loss_standard = train_with_ddp(model_copy_1)

    # Train with UnslothDDP
    loss_optimized = train_with_unsloth_ddp(model_copy_2)

    # Should converge to similar loss
    assert abs(loss_standard - loss_optimized) < tolerance
```

---

## Alternatives Considered

### Alternative 1: Use DeepSpeed ZeRO

```python
from deepspeed import initialize
model, optimizer, _, _ = initialize(model=model, config=ds_config)
```

**Rejected:**
- Adds large dependency
- Different API
- May conflict with Unsloth optimizations

### Alternative 2: Use FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel
model = FullyShardedDataParallel(model)
```

**Rejected:**
- Memory optimization focus, not speed
- Complex configuration
- May interfere with LoRA

### Alternative 3: Gradient Accumulation Only

Increase gradient accumulation to reduce sync frequency.

```python
trainer = UnslothTrainer(
    gradient_accumulation_steps=world_size * 4,
)
```

**Rejected:**
- Increases memory usage
- Doesn't solve efficiency issue
- Just reduces sync count

---

## Open Questions

1. **Compression accuracy impact?**
   - Need extensive testing
   - May vary by model/task
   - Could be user-configurable

2. **NCCL vs custom implementation?**
   - NCCL is highly optimized
   - Custom may not beat it
   - Test both approaches

3. **Tensor parallelism support?**
   - Current focus is data parallelism
   - TP more complex
   - Future consideration

4. **Cloud-specific optimizations?**
   - AWS EFA
   - GCP GPUDirect
   - Different optimal strategies

---

## Success Criteria

- [ ] 90%+ efficiency at 4 GPUs
- [ ] 85%+ efficiency at 8 GPUs
- [ ] No accuracy degradation (same loss curve)
- [ ] Works with existing training scripts
- [ ] Automatic strategy selection works
- [ ] Documentation complete

### Measurable Outcomes

| Metric | Before | After |
|--------|--------|-------|
| 2 GPU efficiency | 85-90% | 95% |
| 4 GPU efficiency | 75-80% | 90% |
| 8 GPU efficiency | 69-75% | 85% |
| Communication overhead | 15-25% | 5-10% |

---

## Required Approvals

- [ ] Multi-GPU testing approval
- [ ] Convergence validation
- [ ] Performance benchmarking

---

## Rollback Strategy

1. Environment variable to use standard DDP
2. No changes to model format
3. Can revert to previous version

---

## References

- PyTorch DDP: https://pytorch.org/docs/stable/notes/ddp.html
- Gradient Compression: https://arxiv.org/abs/1712.01887
- Ring All-Reduce: https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/
- NCCL: https://developer.nvidia.com/nccl

---

*Return to [Prioritization Matrix](./00-prioritization-matrix.md)*
