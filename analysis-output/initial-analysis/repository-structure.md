# Unsloth Repository Structure

**Commit SHA:** `341ce85864d191e4a6b7c447b9167c1faf5e20d3`

---

## Directory Tree with Descriptions

```
/home/user/unsloth/
├── unsloth/                          # Main package (38,757 LOC)
│   ├── __init__.py                   # Entry point, dependency validation
│   ├── chat_templates.py             # Chat templates for 100+ models (3,159 LOC)
│   ├── save.py                       # Model export: GGUF, merged, Hub (2,954 LOC)
│   ├── tokenizer_utils.py            # Tokenizer patching utilities (1,105 LOC)
│   ├── trainer.py                    # UnslothTrainer, training utilities (240 LOC)
│   ├── import_fixes.py               # Import-time dependency patches
│   ├── device_type.py                # Hardware detection (CUDA/HIP/XPU)
│   ├── ollama_template_mappers.py    # Ollama format mappings (81KB)
│   ├── _auto_install.py              # Auto-installation utilities
│   │
│   ├── models/                       # Core model implementations (15,953 LOC)
│   │   ├── __init__.py              # Model exports, version info
│   │   ├── loader.py                # FastLanguageModel, FastVisionModel (1,168 LOC)
│   │   ├── mapper.py                # Model name resolution (1,134 LOC)
│   │   ├── _utils.py                # Shared patches, QAT support (2,308 LOC)
│   │   ├── llama.py                 # FastLlamaModel - base class (3,416 LOC)
│   │   ├── mistral.py               # FastMistralModel
│   │   ├── qwen2.py                 # FastQwen2Model
│   │   ├── qwen3.py                 # FastQwen3Model
│   │   ├── qwen3_moe.py             # FastQwen3MoeModel (MoE support)
│   │   ├── gemma.py                 # FastGemmaModel
│   │   ├── gemma2.py                # FastGemma2Model
│   │   ├── granite.py               # FastGraniteModel
│   │   ├── falcon_h1.py             # FastFalconH1Model
│   │   ├── cohere.py                # FastCohereModel
│   │   ├── llama4.py                # FastLlama4Model
│   │   ├── vision.py                # FastBaseModel, vision support (1,239 LOC)
│   │   ├── dpo.py                   # DPO/KTO trainer patches
│   │   ├── rl.py                    # RL training: GRPO, GSPO, DAPO (1,275 LOC)
│   │   └── rl_replacements.py       # RL component replacements (903 LOC)
│   │
│   ├── kernels/                      # Triton kernels (4,168 LOC)
│   │   ├── __init__.py              # Kernel exports
│   │   ├── cross_entropy_loss.py    # Optimized CE kernel (15KB)
│   │   ├── fast_lora.py             # Fast LoRA implementation (717 LOC)
│   │   ├── flex_attention.py        # Flexible attention mechanisms
│   │   ├── fp8.py                   # FP8 quantization kernel (558 LOC)
│   │   ├── geglu.py                 # GEGLU activation kernel
│   │   ├── layernorm.py             # Layer normalization
│   │   ├── rms_layernorm.py         # RMS LayerNorm kernel (9.7KB)
│   │   ├── rope_embedding.py        # RoPE embedding kernel
│   │   ├── swiglu.py                # SwiGLU activation kernel
│   │   ├── utils.py                 # Kernel utilities (1,002 LOC)
│   │   └── moe/                     # Mixture of Experts
│   │       ├── grouped_gemm/        # Grouped GEMM kernels
│   │       │   ├── interface.py     # MoE interface (968 LOC)
│   │       │   ├── kernels/         # Forward/backward implementations
│   │       │   └── reference/       # Reference implementations
│   │       ├── tests/               # MoE kernel tests
│   │       └── benchmark/           # Performance benchmarking
│   │
│   ├── registry/                     # Model registry (972 LOC)
│   │   ├── __init__.py              # Registry exports
│   │   ├── registry.py              # Core implementation
│   │   ├── _llama.py                # Llama model registry
│   │   ├── _mistral.py              # Mistral model registry
│   │   ├── _qwen.py                 # Qwen model registry
│   │   ├── _gemma.py                # Gemma model registry
│   │   ├── _deepseek.py             # DeepSeek model registry
│   │   ├── _phi.py                  # Phi model registry
│   │   └── REGISTRY.md              # Registry documentation
│   │
│   ├── dataprep/                     # Data preparation
│   │   ├── __init__.py              # Exports
│   │   ├── synthetic.py             # Synthetic data generation
│   │   └── synthetic_configs.py     # Dataset configurations
│   │
│   └── utils/                        # General utilities
│       ├── __init__.py
│       └── hf_hub.py                # HuggingFace Hub utilities
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_model_registry.py       # Registry tests
│   ├── qlora/                       # QLoRA validation tests
│   │   ├── test_hf_qlora_train_and_merge.py
│   │   └── test_unsloth_qlora_train_and_merge.py
│   ├── saving/                      # Model export tests
│   │   ├── language_models/         # LLM tests
│   │   ├── vision_models/           # Vision model tests
│   │   ├── text_to_speech_models/   # TTS tests
│   │   └── non_peft/                # Non-PEFT tests
│   └── utils/                       # Test utilities
│       ├── cleanup_utils.py
│       ├── data_utils.py
│       ├── perplexity_eval.py
│       ├── ocr_eval.py
│       └── aime_eval.py
│
├── scripts/                          # Utility scripts
│   ├── enforce_kwargs_spacing.py    # Custom code style enforcement
│   └── run_ruff_format.py           # Ruff formatting wrapper
│
├── unsloth-cli.py                    # Command-line interface (13.5KB)
├── pyproject.toml                    # Project configuration (60KB)
├── README.md                         # Main documentation (28KB)
├── CONTRIBUTING.md                   # Contribution guidelines
├── CODE_OF_CONDUCT.md               # Community standards
├── .pre-commit-config.yaml          # Pre-commit hooks
└── .github/                          # GitHub configuration
    └── workflows/                    # CI workflows
```

---

## Module Responsibilities

### Core Modules

| Module | LOC | Primary Responsibility |
|--------|-----|------------------------|
| `models/llama.py` | 3,416 | Base model class, attention optimization, LoRA patches |
| `save.py` | 2,954 | All model export formats (GGUF, merged, Hub) |
| `models/_utils.py` | 2,308 | Shared utilities, QAT, gradient checkpointing |
| `chat_templates.py` | 3,159 | Chat format templates for all models |
| `models/rl.py` | 1,275 | Reinforcement learning (GRPO/GSPO/DAPO) |
| `models/vision.py` | 1,239 | Vision and multimodal model support |
| `models/loader.py` | 1,168 | Model loading and auto-detection |
| `models/mapper.py` | 1,134 | Model name resolution |
| `tokenizer_utils.py` | 1,105 | Tokenizer patching and utilities |
| `kernels/utils.py` | 1,002 | Kernel utilities, buffer management |

### Entry Points

| Entry Point | Purpose | Usage |
|-------------|---------|-------|
| `FastLanguageModel.from_pretrained()` | Load optimized model | Primary API |
| `FastLanguageModel.get_peft_model()` | Apply LoRA/QLoRA | After loading |
| `UnslothTrainer` | Training with optimizations | Training loop |
| `unsloth_save_model()` | Export merged model | After training |
| `save_to_gguf()` | Export as GGUF | For inference |
| `unsloth-cli.py` | CLI workflow | Command-line usage |

---

## Code Distribution

### By Module
```
models/        41.2% ████████████████░░░░░░░░░░░░░░ 15,953 LOC
kernels/       10.7% ████░░░░░░░░░░░░░░░░░░░░░░░░░░  4,168 LOC
root package   32.1% ████████████░░░░░░░░░░░░░░░░░░ 12,449 LOC
registry/       2.5% █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    972 LOC
tests/         13.5% █████░░░░░░░░░░░░░░░░░░░░░░░░░  5,215 LOC
```

### By File Type
```
.py files:     100% ██████████████████████████████ 38,757 LOC
```

---

## Key Files by Purpose

### Performance Optimization
- `kernels/fast_lora.py` - LoRA kernel fusion (5-10x speedup)
- `kernels/cross_entropy_loss.py` - Fused CE computation
- `kernels/rms_layernorm.py` - Optimized normalization
- `models/llama.py` - Attention patching

### Memory Management
- `kernels/utils.py` - Global buffer management
- `models/_utils.py` - Gradient checkpointing patches
- `models/llama.py` - KV cache optimization

### Model Support
- `models/loader.py` - Auto-detection and loading
- `models/mapper.py` - Name resolution
- `registry/` - Model metadata

### Export/Save
- `save.py` - All export formats
- `chat_templates.py` - Format templates
- `ollama_template_mappers.py` - Ollama integration

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Build config, dependencies (60KB) |
| `.pre-commit-config.yaml` | Code quality hooks |
| `.pre-commit-ci.yaml` | CI pre-commit config |

---

## Test Organization

```
tests/
├── qlora/           # QLoRA validation (compare HF vs Unsloth)
├── saving/
│   ├── language_models/    # LLM perplexity tests
│   ├── vision_models/      # OCR benchmarks
│   └── text_to_speech_models/  # TTS validation
└── utils/           # Shared test utilities
```

**Testing Approach:**
- Comparison testing (HF baseline vs Unsloth)
- Perplexity validation for model accuracy
- OCR/AIME benchmarks for quality assurance
- Manual GPU testing (no CI automation)

---

*Continue to [Dependency Graph](./dependency-graph.md) for dependency analysis.*
