# Blog 4: Extending and Integrating Unsloth

**Reading Time:** 10 minutes
**Commit:** `341ce85864d191e4a6b7c447b9167c1faf5e20d3`

---

## What You'll Learn

- How to add support for new models
- Integration with training pipelines
- Export formats and deployment options
- Common customization use cases

---

## Introduction

Unsloth's architecture is designed for extension. Whether you want to add support for a new model, customize the training pipeline, or export models for different deployment targets, the codebase provides clear patterns to follow.

In this post, we'll walk through the extension points and provide practical examples for common integration scenarios.

---

## Adding Support for New Models

The most common extension is adding a new model family. Let's walk through the process.

### Step 1: Create the Model Class

Create a new file in `unsloth/models/` for your model:

```python
# unsloth/models/newmodel.py

from .llama import FastLlamaModel, LlamaAttention_fast_forward
from transformers.models.newmodel import modeling_newmodel

class FastNewModel(FastLlamaModel):
    """
    Unsloth optimization for NewModel architecture.
    """

    @staticmethod
    def pre_patch():
        """Apply patches before model loading."""
        # Import the target module
        import transformers.models.newmodel.modeling_newmodel as mm

        # Patch attention
        mm.NewModelAttention.forward = NewModelAttention_fast_forward

        # Patch layer norm (if using RMS norm like Llama)
        mm.NewModelRMSNorm.forward = fast_rms_layernorm_forward

        # Patch decoder layer
        mm.NewModelDecoderLayer.forward = NewModelDecoderLayer_fast_forward


def NewModelAttention_fast_forward(self, hidden_states, **kwargs):
    """
    Optimized attention for NewModel.

    If NewModel uses standard multi-head attention like Llama,
    you can reuse LlamaAttention_fast_forward directly:
    """
    return LlamaAttention_fast_forward(self, hidden_states, **kwargs)
```

### Step 2: Register the Model

Add registry entries in `unsloth/registry/`:

```python
# unsloth/registry/_newmodel.py

from ..models.newmodel import FastNewModel
from .registry import register_model, ModelInfo, QuantType

# Register each variant
register_model("org/newmodel-7b", ModelInfo(
    model_type="newmodel",
    architecture=FastNewModel,
    quantization=QuantType.NONE,
    max_seq_length=4096,
))

register_model("org/newmodel-7b-bnb-4bit", ModelInfo(
    model_type="newmodel",
    architecture=FastNewModel,
    quantization=QuantType.BNB_4BIT,
    max_seq_length=4096,
))
```

### Step 3: Export from Package

Add exports in `unsloth/models/__init__.py`:

```python
from .newmodel import FastNewModel
```

And import the registry in `unsloth/registry/__init__.py`:

```python
from ._newmodel import *
```

### Step 4: Add Chat Template (Optional)

If the model has a specific chat format, add it to [`unsloth/chat_templates.py`](https://github.com/unslothai/unsloth/blob/341ce85864d191e4a6b7c447b9167c1faf5e20d3/unsloth/chat_templates.py):

```python
CHAT_TEMPLATES["newmodel"] = """
{%- for message in messages %}
{%- if message['role'] == 'user' %}
User: {{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{%- endif %}
{%- endfor %}
"""
```

---

## Integration with Training Pipelines

### Using UnslothTrainer

The simplest integration uses `UnslothTrainer` directly:

```python
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Load and prepare dataset
dataset = load_dataset("your/dataset")

# Configure training
args = UnslothTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    embedding_learning_rate=2e-5,  # Separate LR for embeddings
    max_steps=1000,
    fp16=True,
)

# Train
trainer = UnslothTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Integration with Standard TRL

If you prefer TRL's trainers directly, apply Unsloth patches manually:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Load with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, ...)

# Use TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### DPO Training

For Direct Preference Optimization:

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Apply Unsloth patches to DPOTrainer
PatchDPOTrainer()

# Create trainer
trainer = DPOTrainer(
    model=model,
    args=DPOConfig(...),
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Reinforcement Learning

For GRPO and other RL methods:

```python
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# Apply RL patches
PatchFastRL()

# Load and configure
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Train with GRPO
trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(...),
    train_dataset=dataset,
)
```

---

## Export and Deployment

### Export to GGUF

GGUF is the standard format for llama.cpp and Ollama:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("your-trained-model")

# Export to GGUF with quantization
model.save_pretrained_gguf(
    "output-gguf",
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit quantization
)
```

Available quantization methods:
- `q4_0`, `q4_1`, `q4_k_m`, `q4_k_s` - 4-bit variants
- `q5_0`, `q5_1`, `q5_k_m`, `q5_k_s` - 5-bit variants
- `q8_0` - 8-bit
- `f16` - 16-bit float

### Save Merged Model

Merge LoRA weights back into base model:

```python
# Merge and save locally
model.save_pretrained_merged(
    "merged-model",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit"
)
```

### Push to HuggingFace Hub

```python
# Push merged model
model.push_to_hub_merged(
    "your-username/model-name",
    tokenizer,
    save_method="merged_16bit",
)

# Push GGUF
model.push_to_hub_gguf(
    "your-username/model-name-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Create Ollama Model

```python
# Save as GGUF with Ollama modelfile
model.save_pretrained_gguf(
    "ollama-model",
    tokenizer,
    quantization_method="q4_k_m",
)

# Generated Modelfile can be used with:
# ollama create mymodel -f ollama-model/Modelfile
```

---

## Common Customization Patterns

### Custom LoRA Targets

Target specific modules for LoRA:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
        "embed_tokens", "lm_head",                # Embeddings
    ],
    lora_alpha=32,
    lora_dropout=0.05,
)
```

### Custom Learning Rate Schedule

```python
from transformers import get_cosine_schedule_with_warmup

# After creating trainer
num_training_steps = len(train_dataset) // batch_size * num_epochs

scheduler = get_cosine_schedule_with_warmup(
    trainer.optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

trainer.lr_scheduler = scheduler
```

### Mixed Precision Configuration

```python
args = UnslothTrainingArguments(
    fp16=True,          # Use FP16
    # or
    bf16=True,          # Use BF16 (better for newer GPUs)

    # For 4-bit training
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
)
```

### Gradient Accumulation for Large Batches

```python
# Effective batch size = per_device_batch_size * gradient_accumulation_steps * num_gpus
args = UnslothTrainingArguments(
    per_device_train_batch_size=1,      # Small for memory
    gradient_accumulation_steps=32,      # Large effective batch
)
```

---

## Custom Kernels Integration

### Using Custom Triton Kernels

If you've written custom Triton kernels:

```python
import triton
import triton.language as tl

@triton.jit
def custom_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = custom_operation(x)
    tl.store(output_ptr + offsets, output, mask=mask)

# Use in your model
class CustomLayer(nn.Module):
    def forward(self, x):
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        custom_kernel[grid](x, output, x.numel(), BLOCK_SIZE=1024)
        return output
```

### Replacing Built-in Kernels

To replace Unsloth's kernels with your own:

```python
# Override before loading model
import unsloth.kernels.rms_layernorm as rms_module

def my_fast_rms_layernorm(hidden_states, weight, eps):
    # Your optimized implementation
    ...

rms_module.fast_rms_layernorm = my_fast_rms_layernorm
```

---

## Debugging Integration Issues

### Verify Optimizations Applied

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)

# Check attention is patched
attention = model.model.layers[0].self_attn
print(attention.forward.__name__)  # Should include "fast"

# Check memory usage
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Common Issues

1. **"Model not supported" error:**
```python
# Check model_type in config
from transformers import AutoConfig
config = AutoConfig.from_pretrained("model-name")
print(config.model_type)  # Must be in SUPPORTED_MODELS
```

2. **No speedup observed:**
```python
# Ensure unsloth imported first
import unsloth  # MUST be before transformers
from transformers import ...
```

3. **CUDA out of memory:**
```python
# Try these in order:
# 1. Reduce batch size
# 2. Enable gradient checkpointing (default)
# 3. Use 4-bit quantization
# 4. Reduce sequence length
```

---

## Production Deployment Checklist

- [ ] Merge LoRA weights for inference speed
- [ ] Quantize for deployment hardware (GGUF for CPU, INT8 for GPU)
- [ ] Test on target hardware
- [ ] Validate output quality (perplexity, task metrics)
- [ ] Set up monitoring for inference latency
- [ ] Configure appropriate batch sizes for hardware

---

## Key Takeaways

1. **Adding new models** follows a clear pattern: create class, register, export

2. **Multiple training integrations** are supported: UnslothTrainer, TRL, DPO, RL

3. **Export options** cover all major deployment targets: GGUF, merged, Hub

4. **Customization is straightforward** through LoRA config, training args, and kernel replacement

5. **Debugging tools** exist to verify optimizations are applied

---

## What's Next

In [Blog 5: Performance Analysis](./05-performance-analysis.md), we'll dive deep into Unsloth's performance characteristics, analyze bottlenecks, and explore optimization opportunities.

---

*Continue to [Blog 5: Performance Analysis and Optimization Opportunities](./05-performance-analysis.md)*
