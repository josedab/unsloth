# Unsloth Terminology Glossary

**Commit SHA:** `341ce85864d191e4a6b7c447b9167c1faf5e20d3`

---

## Core Concepts

### FastLanguageModel
The primary user-facing class for loading and optimizing language models. Automatically detects model architecture and applies appropriate optimizations.

**Usage:**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
```

### FastVisionModel
Variant of FastLanguageModel for vision and multimodal models (LLaVA, Qwen-VL, etc.).

### FastLlamaModel
The base class for all model optimizations. Despite the name, it serves as the parent class for all architectures (Mistral, Qwen, Gemma, etc.).

### UnslothTrainer
Extended version of TRL's SFTTrainer with Unsloth-specific optimizations like embedding-specific learning rates and gradient accumulation fixes.

---

## Optimization Techniques

### LoRA (Low-Rank Adaptation)
Parameter-efficient fine-tuning technique that trains small adapter matrices instead of full model weights. Unsloth provides optimized LoRA implementations with 5-10x faster backward passes.

**Related:** QLoRA, Fast LoRA

### QLoRA (Quantized LoRA)
LoRA applied to quantized (4-bit or 8-bit) base models. Enables fine-tuning of large models on consumer GPUs.

### Gradient Checkpointing
Memory optimization technique that recomputes intermediate activations during backward pass instead of storing them. Unsloth's "smart" gradient checkpointing selectively applies this to maximize memory savings with minimal performance impact.

### Flash Attention
Memory-efficient attention algorithm that reduces attention complexity from O(n²) to O(n) memory. Unsloth integrates flash-attn for supported models.

### Triton Kernels
Custom CUDA kernels written in Triton (OpenAI's GPU programming language). Unsloth uses Triton for:
- Fast LoRA operations
- Cross-entropy loss computation
- Layer normalization
- Activation functions (SwiGLU, GEGLU)

---

## Quantization

### 4-bit Quantization
Compression technique that stores model weights in 4 bits instead of 16/32 bits. Reduces VRAM usage by 75% with minimal quality loss.

### 8-bit Quantization
Less aggressive compression storing weights in 8 bits. Better quality but less memory savings than 4-bit.

### QAT (Quantization-Aware Training)
Training technique that simulates quantization during training, resulting in better quantized model quality.

### FP8 (8-bit Floating Point)
New floating-point format optimized for ML workloads. Supported on newer GPUs (H100, RTX 4090).

### bitsandbytes
Library providing 4-bit and 8-bit quantization for PyTorch models. Unsloth integrates deeply with bitsandbytes for QLoRA.

---

## Training Methods

### SFT (Supervised Fine-Tuning)
Standard fine-tuning on instruction/response pairs. The primary training method supported by Unsloth.

### DPO (Direct Preference Optimization)
Alignment technique that trains models on preference data (chosen vs rejected responses). Unsloth patches TRL's DPOTrainer.

### GRPO (Group Relative Policy Optimization)
Reinforcement learning method for LLM alignment. Supported via `PatchFastRL`.

### GSPO (Group Supervised Policy Optimization)
Variant of GRPO with supervised elements.

### DAPO (Direct Advantage Policy Optimization)
Another RL variant supported by Unsloth.

### KTO (Kahneman-Tversky Optimization)
Alignment method based on prospect theory. Supported via `PatchKTOTrainer`.

---

## Model Components

### RoPE (Rotary Position Embedding)
Position encoding method used by Llama, Mistral, and other modern LLMs. Unsloth provides optimized RoPE kernels.

### RMS LayerNorm (Root Mean Square Layer Normalization)
Simplified normalization used by Llama-family models. Unsloth provides Triton-optimized implementation.

### SwiGLU / GEGLU
Activation functions used in modern LLMs. Unsloth provides fused kernel implementations.

### KV Cache (Key-Value Cache)
Stores computed key-value pairs during inference to avoid recomputation. Unsloth optimizes KV cache allocation.

### MoE (Mixture of Experts)
Architecture using multiple "expert" networks with a routing mechanism. Unsloth supports MoE models like Qwen3-MoE with optimized grouped GEMM kernels.

---

## Export Formats

### GGUF (GPT-Generated Unified Format)
Binary format for LLM inference, used by llama.cpp, Ollama, and other inference engines. Unsloth provides `save_to_gguf()` for direct export.

### Merged Model
A model where LoRA adapter weights have been merged back into the base model weights. Created with `merge_and_unload()`.

### Sharded Model
Large model split across multiple files for easier distribution. Unsloth handles sharding automatically during export.

---

## Architecture Terms

### Monkey-Patching
Unsloth's technique of replacing methods in HuggingFace/PEFT objects at runtime with optimized versions. Enables drop-in performance improvements without forking.

### Decorator Pattern
Design pattern where Unsloth wraps existing functionality with additional optimizations. Used throughout the codebase.

### Model Registry
System for mapping model names to their implementations. Located in `unsloth/registry/`.

### Model Mapper
Component that resolves model names (including aliases and versions) to canonical forms. Located in `models/mapper.py`.

---

## Hardware Terms

### Compute Capability
NVIDIA's classification of GPU features. Unsloth requires CC 7.0+ (V100 and newer).

### CUDA
NVIDIA's parallel computing platform. Core requirement for Unsloth.

### ROCm
AMD's GPU computing platform. Partially supported by Unsloth.

### XPU
Intel's discrete GPU platform. Partially supported.

### VRAM (Video RAM)
GPU memory. Unsloth's primary optimization target is reducing VRAM usage.

---

## HuggingFace Ecosystem

### transformers
HuggingFace's model library. Unsloth patches and extends transformers models.

### PEFT (Parameter-Efficient Fine-Tuning)
HuggingFace library for LoRA and other efficient fine-tuning methods. Unsloth integrates deeply with PEFT.

### TRL (Transformer Reinforcement Learning)
HuggingFace library for training LLMs with RL. Provides SFTTrainer that Unsloth extends.

### accelerate
HuggingFace library for distributed training. Used by Unsloth for device management.

### datasets
HuggingFace data loading library. Used by Unsloth training workflows.

### huggingface_hub
Library for interacting with HuggingFace model hub. Used for model download and upload.

---

## Code Patterns

### `pre_patch()`
Method in model classes that applies optimizations before model loading. Patches attention, normalization, and other components.

### `post_patch()`
Method that applies optimizations after model loading.

### `for_inference()`
Method to prepare a trained model for inference mode. Disables gradient computation, optimizes KV cache.

### `get_peft_model()`
Method to apply LoRA configuration to a loaded model. Returns a PEFT-wrapped model.

### `patch_*` functions
Functions that modify external library behavior. E.g., `patch_unsloth_smart_gradient_checkpointing()`.

---

## Configuration

### LoraConfig
PEFT configuration for LoRA training:
- `r`: Rank of LoRA matrices (typically 8-64)
- `lora_alpha`: Scaling factor (typically 16-32)
- `target_modules`: Which layers to apply LoRA
- `lora_dropout`: Dropout for regularization

### TrainingArguments
Configuration for training:
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Steps before weight update
- `learning_rate`: Optimizer learning rate
- `max_steps`: Total training steps

### UnslothTrainingArguments
Extended TrainingArguments with:
- `embedding_learning_rate`: Separate LR for embeddings

---

## File Locations

| Term | Location |
|------|----------|
| FastLanguageModel | `unsloth/models/loader.py` |
| FastLlamaModel | `unsloth/models/llama.py` |
| UnslothTrainer | `unsloth/trainer.py` |
| Triton kernels | `unsloth/kernels/` |
| Model registry | `unsloth/registry/` |
| Chat templates | `unsloth/chat_templates.py` |
| Save functions | `unsloth/save.py` |

---

*Return to [Quick Start](./00-quick-start.md) for the analysis overview.*
