# Unsloth Architectural Patterns - Quick Reference

## File Locations & Pattern Usage

### 1. FACTORY PATTERN
**Location**: `/home/user/unsloth/unsloth/models/loader.py` (lines 115-400+)

```
FastLanguageModel.from_pretrained()  [Main factory method]
    ├─ Routes to: FastLlamaModel, FastMistralModel, FastQwen2Model, etc.
    └─ Auto-detects model type from config
```

**Related Files**:
- `models/loader_utils.py` - Model name mapping utilities
- `models/mapper.py` - Quantization variant mapping

**Usage Pattern**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True
)
```

---

### 2. DECORATOR PATTERN (Monkey-Patching)
**Location**: `/home/user/unsloth/unsloth/models/llama.py` (lines 2055-2091)

```
FastLlamaModel.pre_patch()  [Class-level patching]
    ├─ Patches: LlamaAttention.forward
    ├─ Patches: LlamaSdpaAttention.forward
    ├─ Patches: LlamaFlashAttention2.forward
    ├─ Patches: LlamaDecoderLayer.forward
    ├─ Patches: LlamaModel.forward
    ├─ Patches: LlamaForCausalLM.forward
    └─ Patches: PeftModelForCausalLM.forward
```

**Related Patching Layers**:
- `import_fixes.py` (lines 100+) - Global library patches
- `models/_utils.py` (lines 507+) - Utility patches
- `trainer.py` - Training-specific patches

**Key Patching Functions**:
- `patch_model_and_tokenizer()` - Post-load patches
- `patch_gradient_checkpointing()` - Training optimization
- `patch_hf_quantizer()` - Quantization setup

---

### 3. STRATEGY PATTERN
**Location**: `/home/user/unsloth/unsloth/models/loader.py` (lines 190-230)

**Quantization Strategies**:
```
load_in_4bit=True
    └─ BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

load_in_8bit=True
    └─ prepare_model_for_kbit_training()

qat_scheme=Int4WeightOnlyConfig()
    └─ _prepare_model_for_qat()

None (full precision)
    └─ AutoModelForCausalLM.from_pretrained()
```

**Attention Strategies**:
Located in `/home/user/unsloth/unsloth/models/llama.py` (lines 528-650+)
```
if HAS_FLASH_ATTENTION:
    output = flash_attn_func()  [Best performance]
elif HAS_XFORMERS:
    output = xformers_attention()  [Good balance]
else:
    output = scaled_dot_product_attention()  [Fallback]
```

**Kernel Selection**:
Located in `/home/user/unsloth/unsloth/kernels/`
```
- fast_lora.py (717 LOC)
- cross_entropy_loss.py (459 LOC)
- rms_layernorm.py (335 LOC)
- rope_embedding.py (239 LOC)
```

---

### 4. TEMPLATE METHOD PATTERN
**Location**: `/home/user/unsloth/unsloth/models/llama.py` (lines 2048+) [Base]

**Model Implementations**:
```
FastLlamaModel  [Base template]
    ├── FastMistralModel (mistral.py)
    ├── FastGemmaModel (gemma.py)
    ├── FastGemma2Model (gemma2.py)
    ├── FastQwen2Model (qwen2.py)
    ├── FastQwen3Model (qwen3.py)
    ├── FastQwen3MoeModel (qwen3_moe.py)
    ├── FastGraniteModel (granite.py)
    ├── FastFalconH1Model (falcon_h1.py)
    └── ... (20+ other models)
```

**Template Methods** (overridable):
1. `pre_patch()` - Model-specific architecture patching
2. `from_pretrained()` - Custom loading logic
3. `_prepare_for_qat()` - QAT-specific setup

---

### 5. REGISTRY PATTERN
**Location**: `/home/user/unsloth/unsloth/registry/registry.py`

**Core Classes**:
```
ModelInfo         [Model metadata dataclass]
    ├─ org, base_name, version, size
    ├─ is_multimodal, quant_type
    └─ model_path property

QuantType         [Enum for quantization types]
    ├─ BNB (BitsAndBytes)
    ├─ UNSLOTH (Dynamic 4-bit)
    ├─ GGUF (llama.cpp format)
    ├─ NONE (Full precision)
    └─ BF16 (16-bit only)

MODEL_REGISTRY    [Central registry dict]
    ├─ Key: "{org}/{name}"
    └─ Value: ModelInfo instance
```

**Registration**:
Located in `/home/user/unsloth/unsloth/registry/_llama.py`, `_mistral.py`, etc.
```python
register_model(
    model_info_cls=LlamaModelInfo,
    org="unsloth",
    base_name="llama-3",
    version="8b",
    size=8000000000,
    quant_type=QuantType.BNB
)
```

**Model Variants**:
Located in `/home/user/unsloth/unsloth/models/mapper.py`
```
INT_TO_FLOAT_MAPPER  [Maps quantized -> float versions]
FLOAT_TO_INT_MAPPER  [Maps float -> quantized versions]
```

---

### 6. ADAPTER PATTERN
**Location**: `/home/user/unsloth/unsloth/kernels/fast_lora.py` (717 LOC)

**LoRA Kernel Adaptations**:
```
apply_lora_qkv()           [QKV projection]
apply_lora_o()             [Output projection]
apply_lora_mlp_swiglu()    [SWIGLU MLP]
apply_lora_mlp_geglu_exact() [GEGLU MLP]
apply_lora_mlp_geglu_approx() [Approximate GEGLU]
fast_lora_forward()        [Unified LoRA forward]
```

**Patching**:
Located in `/home/user/unsloth/unsloth/models/_utils.py` (lines 1798+)
```python
def patch_fast_lora():
    # Replace PEFT LoRA forward with fast kernel version
    Linear.forward = fast_lora_forward
    Linear4bit.forward = fast_lora_forward
```

---

## Core Abstractions & Relationships

### Class Hierarchy
```
FastBaseModel (Abstract base in vision.py)
    ├─ from_pretrained() [Static factory]
    ├─ for_inference() [Mode switching]
    ├─ for_training() [Mode switching]
    └─ get_peft_model() [LoRA wrapping]

    └─ FastLlamaModel (Concrete implementation in llama.py)
        ├─ pre_patch() [Architecture patching]
        ├─ from_pretrained() [Loading logic]
        └─ _prepare_for_qat() [QAT setup]
        
        └─ Model-specific subclasses
            ├─ FastMistralModel
            ├─ FastGemmaModel
            ├─ FastQwen2Model
            └─ ... (17+ other models)
```

### Dependency Injection Pattern
```
FastLanguageModel.from_pretrained(
    model_patcher=None  # Defaults to FastLlamaModel
)
```
Allows injecting custom model patcher implementations

### Mixin-like Behavior
```
All Fast*Model classes inherit:
    ├─ FastLlamaModel methods (pre_patch, from_pretrained)
    └─ FastBaseModel methods (for_inference, for_training, get_peft_model)
```

---

## ML-Specific Patterns

### 1. TRAINING MODE SWITCHING
**Location**: `/home/user/unsloth/unsloth/models/vision.py` (lines 1138-1210)

```python
# For inference
FastBaseModel.for_inference(model)
    ├─ Sets use_cache=True
    ├─ Sets eval() mode
    └─ Enables torch.compile

# For training
FastBaseModel.for_training(model)
    ├─ Sets use_cache=False (incompatible with gradient checkpointing)
    ├─ Sets train() mode
    └─ Patches gradient checkpointing
```

### 2. GRADIENT CHECKPOINTING STRATEGY
**Location**: Imported from `unsloth_zoo.gradient_checkpointing`

**Three variants**:
1. `"unsloth"` (default) - Smart selection of expensive layers only
2. Custom offloaded checkpoint - Offload to disk/CPU
3. `True` - Standard transformers gradient checkpointing

**Setup**:
```python
patch_unsloth_smart_gradient_checkpointing(
    model,
    use_reentrant=False  # Unsloth specific
)
```

### 3. QUANTIZATION PROGRESSION
```
Full Precision (torch.float32 or torch.float16)
    ↓ Higher Memory
    ↓
Full Finetuning (No quantization)
    ↓
8-bit (PEFT 8-bit quantization)
    ↓
QAT (Quantize-Aware Training)
    ↓
4-bit (BitsAndBytes with NF4)
    ↓ Lower Memory
```

### 4. KV CACHE MANAGEMENT
**Location**: `/home/user/unsloth/unsloth/models/llama.py` (lines 280-350)

```
Training: use_cache=False
    └─ Incompatible with gradient checkpointing

Inference: use_cache=True
    └─ Incremental cache (KV_CACHE_INCREMENT=512)
    └─ Reduces memory pressure during generation
```

---

## Memory Optimization Hierarchy

```
No Optimization (Full Precision)
    ├─ Gradient Checkpointing (30-50% savings)
    ├─ Smart Gradient Checkpointing (40-50% savings)
    ├─ Embedding Offloading (200MB-1GB savings)
    ├─ Offloaded Gradient Checkpoint (50-70% savings)
    └─ Dynamic KV Cache (10-20% savings)
    
4-bit Quantization (75% savings)
    ├─ BitsAndBytes 4-bit
    ├─ Combined with gradient checkpointing
    └─ ~6-8GB VRAM for 7B models
    
8-bit Quantization (50% savings)
    ├─ PEFT 8-bit
    └─ ~10-12GB VRAM for 7B models
```

---

## Extension Points Summary

### Add New Model Type
**Files to modify**:
1. Create `/home/user/unsloth/unsloth/models/mymodel.py`
   - Inherit FastLlamaModel
   - Override pre_patch()
   - Override from_pretrained()

2. Register in `/home/user/unsloth/unsloth/registry/_mymodel.py`
   - Create ModelInfo subclass
   - Call register_model()

3. Update `/home/user/unsloth/unsloth/models/loader.py`
   - Add dispatch logic in from_pretrained()

### Add Custom Kernel
**Files**:
1. Create `/home/user/unsloth/unsloth/kernels/my_kernel.py`
   - Write Triton @jit kernel
   - Wrap in forward function
   - Export in __init__.py

2. Patch module in appropriate model class
   - MyModule.forward = my_kernel_forward

### Add Training Optimization
**Files**:
1. Update `/home/user/unsloth/unsloth/trainer.py`
   - Add UnslothTrainingArguments field
   - Add optimizer modification logic

2. Or patch TRL trainer in `/home/user/unsloth/unsloth/models/dpo.py`, `rl.py`
   - Override trainer methods
   - Inject Unsloth optimizations

---

## Performance Measurement Points

```
Pre-optimization:
    ├─ HAS_FLASH_ATTENTION (fastest attention)
    ├─ HAS_XFORMERS (medium attention)
    ├─ SDPA_HAS_GQA (supports grouped query attention)
    └─ HAS_CUTLASS (GEMM optimization)

Post-optimization:
    ├─ Training speedup: 2-5x
    ├─ Inference speedup: 2-3x
    ├─ Memory savings: 40-90%
    └─ LoRA overhead: ~10-15% vs full weights
```

---

## Critical Version Checks

Located throughout codebase:
- `transformers >= 4.37` (4-bit support)
- `transformers >= 4.46` (Granite support)
- `transformers >= 4.50.3` (Qwen3 support)
- `torch >= 2.0` (torch.compile support)
- `triton >= 3.0` (updated driver interface)

**Version-specific Branching**:
- Gradient accumulation fix (transformers <= 4.45.2)
- torch.cuda.amp compatibility (torch < 2.4)
- Rope scaling parameter injection (transformers < 4.46)

