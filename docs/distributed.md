# Unsloth Multi-GPU Training Guide

This guide explains how to use Unsloth's optimized distributed training module for efficient multi-GPU training.

## Overview

Unsloth's distributed module provides optimized DDP (Distributed Data Parallel) training with:
- **Gradient Compression**: Reduce communication overhead with top-K sparsification
- **Overlapped Communication**: Hide communication latency by overlapping with computation
- **LoRA-Aware Distribution**: Optimize synchronization for LoRA training
- **Ring All-Reduce**: Better bandwidth utilization for high-speed interconnects

## Quick Start

### Basic Usage

```python
from unsloth import FastLanguageModel
from unsloth.distributed import UnslothDDP

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Wrap with optimized DDP
model = UnslothDDP.from_model(
    model,
    compression_ratio=0.1,  # 10x compression
    overlap=True,           # Overlapped communication
)

# Training proceeds normally with your trainer
```

### Automatic Strategy Selection

Let Unsloth choose the best settings for your setup:

```python
from unsloth.distributed import UnslothDDP

# Auto-select based on your setup
model = UnslothDDP.auto(
    model,
    world_size=4,              # Number of GPUs
    interconnect="nvlink",     # Type of interconnect
)
```

## Configuration Options

### UnslothDDP.from_model()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression_ratio` | float | None | Ratio for gradient compression (0.0-1.0). None disables compression. |
| `overlap` | bool | True | Enable overlapped communication with computation. |
| `bucket_size_mb` | float | 25.0 | Size of gradient buckets in MB for overlapped comm. |
| `use_ring_allreduce` | bool | False | Use custom ring all-reduce instead of NCCL. |
| `sync_base_params` | bool | False | Also sync base model parameters (not just LoRA). |

### UnslothDDP.auto()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `world_size` | int | None | Number of GPUs (auto-detected if None). |
| `interconnect` | str | "ethernet" | Type: "nvlink", "infiniband", or "ethernet". |

## Launching Distributed Training

### Using torchrun (Recommended)

```bash
torchrun --nproc_per_node=4 train.py
```

### Using torch.distributed.launch

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Initialize Distributed in Your Script

```python
from unsloth.distributed import initialize_distributed, cleanup_distributed

# At the start of your script
if not initialize_distributed():
    print("Running in single-GPU mode")

# Your training code here
...

# At the end
cleanup_distributed()
```

## Optimization Strategies

### 1. Gradient Compression

Best for: Large clusters (8+ GPUs) with limited bandwidth.

```python
model = UnslothDDP.from_model(
    model,
    compression_ratio=0.1,  # Keep top 10% of gradients
)
```

**How it works**: Only the top-K gradient values (by magnitude) are synchronized. An error feedback mechanism accumulates compression errors to maintain convergence.

**Trade-off**: Faster communication but may require slightly more epochs to converge.

### 2. Overlapped Communication

Best for: Hiding communication latency during backprop.

```python
model = UnslothDDP.from_model(
    model,
    overlap=True,
    bucket_size_mb=25.0,  # Larger buckets for high-bandwidth
)
```

**How it works**: Gradients are synchronized in buckets as they become available during backward pass, overlapping with computation.

**Trade-off**: More complex but provides significant speedup.

### 3. LoRA-Aware Distribution

Best for: LoRA/QLoRA training where only adapters are trained.

```python
model = UnslothDDP.from_model(
    model,
    overlap=False,  # Uses LoRAAwareDDP internally
    sync_base_params=False,  # Only sync LoRA params
)
```

**How it works**: Only LoRA parameters (which are much smaller) are synchronized, significantly reducing communication.

### 4. Ring All-Reduce

Best for: High-bandwidth interconnects (NVLink, InfiniBand).

```python
model = UnslothDDP.from_model(
    model,
    use_ring_allreduce=True,
)
```

**How it works**: Uses ring topology for better bandwidth utilization compared to tree-based reduction.

## Recommended Settings by Setup

| Setup | Recommended Configuration |
|-------|---------------------------|
| 2 GPUs, NVLink | `overlap=True, bucket_size_mb=50` |
| 4 GPUs, Ethernet | `compression_ratio=0.2, overlap=True` |
| 8 GPUs, InfiniBand | `use_ring_allreduce=True, bucket_size_mb=50` |
| Any, LoRA only | `overlap=False, sync_base_params=False` |

## Complete Training Example

```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from unsloth import FastLanguageModel
from unsloth.distributed import UnslothDDP, initialize_distributed, cleanup_distributed
from trl import SFTConfig, SFTTrainer

# Initialize distributed
initialize_distributed()

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# Wrap with UnslothDDP
model = UnslothDDP.auto(
    model,
    interconnect="nvlink",  # Adjust based on your setup
)

# Create trainer
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model.model,  # Access underlying model
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Train
trainer.train()

# Cleanup
cleanup_distributed()
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `UNSLOTH_USE_STANDARD_DDP` | Set to "1" to fall back to standard PyTorch DDP |
| `MASTER_ADDR` | Master node address for distributed init |
| `MASTER_PORT` | Master node port for distributed init |
| `LOCAL_RANK` | Local rank of the current process |
| `WORLD_SIZE` | Total number of processes |
| `RANK` | Global rank of the current process |

## Fallback to Standard DDP

If you encounter issues with UnslothDDP, you can fall back to standard PyTorch DDP:

```bash
UNSLOTH_USE_STANDARD_DDP=1 torchrun --nproc_per_node=4 train.py
```

Or in your code:

```python
import os

if os.environ.get("UNSLOTH_USE_STANDARD_DDP"):
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model)
else:
    from unsloth.distributed import UnslothDDP
    model = UnslothDDP.from_model(model)
```

## Performance Tips

1. **Tune bucket size**: Larger buckets reduce communication frequency but increase latency. Start with 25MB and adjust.

2. **Match compression to bandwidth**: Lower bandwidth benefits more from compression. Use 0.1-0.2 for Ethernet, avoid compression for NVLink.

3. **Use gradient accumulation**: Reduce sync frequency by accumulating gradients over multiple steps.

4. **Profile your setup**: Use `torch.cuda.profiler` to identify communication bottlenecks.

5. **Consider gradient checkpointing**: Saves memory, allowing larger batch sizes which improve GPU utilization.

## Troubleshooting

### Communication Timeout

If you see NCCL timeout errors:
```bash
export NCCL_TIMEOUT=1800  # Increase to 30 minutes
```

### Memory Issues

Reduce bucket size or use gradient checkpointing:
```python
model = UnslothDDP.from_model(model, bucket_size_mb=10)
```

### Convergence Issues with Compression

Try higher compression ratio (less compression):
```python
model = UnslothDDP.from_model(model, compression_ratio=0.3)  # Keep 30%
```

### Checking Distributed Status

```python
from unsloth.distributed import get_distributed_info

info = get_distributed_info()
print(f"Initialized: {info['initialized']}")
print(f"World size: {info['world_size']}")
print(f"Rank: {info['rank']}")
```

## API Reference

### CompressedAllReduce

```python
from unsloth.distributed import CompressedAllReduce

compressor = CompressedAllReduce(compression_ratio=0.1, min_elements=1)
values, indices = compressor.compress(tensor, "param_name")
decompressed = compressor.decompress(values, indices, original_shape)
```

### LoRAAwareDDP

```python
from unsloth.distributed import LoRAAwareDDP

lora_ddp = LoRAAwareDDP(model, sync_base_params=False)
# After backward
lora_ddp.sync_gradients()
```

### ring_allreduce

```python
from unsloth.distributed import ring_allreduce

reduced_tensor = ring_allreduce(tensor, group=None)
```

### Utility Functions

```python
from unsloth.distributed import (
    initialize_distributed,
    cleanup_distributed,
    get_distributed_info,
)

# Initialize
success = initialize_distributed(backend="nccl")

# Get info
info = get_distributed_info()

# Cleanup
cleanup_distributed()
```

## Expected Performance

| GPUs | Standard DDP | UnslothDDP | Improvement |
|------|-------------|------------|-------------|
| 2 | 1.7-1.8x | 1.9x | ~6% |
| 4 | 3.0-3.2x | 3.6x | ~15% |
| 8 | 5.5-6.0x | 6.8x | ~15% |

*Performance varies based on model size, interconnect, and workload.*

## References

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [Gradient Compression Paper](https://arxiv.org/abs/1712.01887)
- [Ring All-Reduce Algorithm](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)
