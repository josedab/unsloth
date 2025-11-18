# Unsloth Dependency Analysis

**Commit SHA:** `341ce85864d191e4a6b7c447b9167c1faf5e20d3`

---

## Dependency Philosophy

Unsloth employs a **zero mandatory dependencies** strategy - users install PyTorch separately based on their CUDA version. This prevents dependency conflicts and gives users maximum flexibility.

---

## Core Dependencies

### Required (When Using Unsloth)

| Package | Version Constraint | Purpose | Last Updated |
|---------|-------------------|---------|--------------|
| **transformers** | >=4.51.3,<=4.57.2 (excl. 4.52.0-4.57.0) | Model loading, tokenizers | Active |
| **torch** | User-managed | Core tensor operations | Active |
| **peft** | >=0.7.1 (excl. 0.14.0) | LoRA/QLoRA implementation | Active |
| **accelerate** | >=0.34.1 | Distributed training | Active |
| **trl** | >=0.18.2,<=0.24.0 | Training framework | Active |
| **datasets** | Any | Data loading | Active |
| **huggingface_hub** | Any | Model hub integration | Active |

### Performance Dependencies

| Package | Version Constraint | Purpose | Platform |
|---------|-------------------|---------|----------|
| **triton** | >=3.0.0 | Custom CUDA kernels | Linux only |
| **xformers** | 0.0.22-0.0.31 | Memory-efficient attention | CUDA |
| **flash-attn** | >=2.6.3 | Flash Attention | Linux/CUDA |
| **bitsandbytes** | >=0.45.5 (excl. 0.46.0, 0.48.0) | 4/8-bit quantization | All |

### Companion Package

| Package | Purpose |
|---------|---------|
| **unsloth_zoo** | Benchmarking, additional utilities |

---

## Version Exclusions (Active Compatibility Management)

Unsloth explicitly excludes specific versions with known issues:

### transformers
- `4.52.0`, `4.52.1`, `4.52.2`, `4.52.3`, `4.52.4`
- `4.53.0`, `4.53.1`
- `4.55.0`, `4.55.1`
- `4.56.0`
- `4.57.0`

### bitsandbytes
- `0.46.0` - Known quantization issues
- `0.48.0` - Compatibility problems

### peft
- `0.14.0` - LoRA bugs

---

## Installation Profiles (50+ Variants)

### By Platform

```toml
# Linux with Triton
pip install unsloth[triton]

# Full HuggingFace ecosystem
pip install unsloth[huggingface]

# Windows
pip install unsloth[windows]
```

### By CUDA Version

```toml
# CUDA 11.8
pip install unsloth[cu118only]

# CUDA 12.1
pip install unsloth[cu121only]

# CUDA 12.4
pip install unsloth[cu124only]
```

### XFormers Variants (70+ wheels)

Unsloth provides pre-built xformers wheels for specific CUDA/PyTorch combinations:

```
xformers-0.0.28.post3+cu121.torch2.5-cp310-linux_x86_64.whl
xformers-0.0.28.post3+cu124.torch2.6-cp311-linux_x86_64.whl
...
```

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Unsloth                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐       │
│  │ FastModel   │  │ UnslothTrainer│  │ save_to_gguf │       │
│  └─────────────┘  └──────────────┘  └───────────────┘       │
└─────────────────────────────────────────────────────────────┘
          │                   │                    │
          ▼                   ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  transformers   │  │      trl        │  │ huggingface_hub │
│  (model arch)   │  │  (SFTTrainer)   │  │  (model export) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│      peft       │  │   accelerate    │
│  (LoRA/QLoRA)   │  │ (distributed)   │
└─────────────────┘  └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Performance Layer                         │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐  │
│  │ triton  │  │ xformers │  │ flash-attn │  │bitsandbytes│  │
│  └─────────┘  └──────────┘  └────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        PyTorch                               │
│              (User-managed, CUDA-specific)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Security Analysis

### CVE Status
- **No known CVEs** in direct Unsloth dependencies
- All core libraries are actively maintained

### License Compatibility
- **Apache-2.0** (Unsloth)
- Compatible with: transformers (Apache-2.0), PyTorch (BSD), peft (Apache-2.0)

### Recommendations
1. **Run `pip-audit` regularly** to check for new vulnerabilities
2. **Monitor transformers releases** - frequent Unsloth incompatibilities
3. **Keep bitsandbytes updated** - active development, occasional bugs

---

## Version Pinning Strategy

### Strict Pinning
- **TRL**: `>=0.18.2,<=0.24.0` - API stability critical
- **xformers**: Specific version per CUDA/PyTorch combo

### Flexible with Exclusions
- **transformers**: Wide range with explicit exclusions
- **bitsandbytes**: Latest with known-bad versions excluded

### User-Managed
- **PyTorch**: No version specified - user controls CUDA compatibility
- **CUDA Toolkit**: System-level, not Python-managed

---

## Dependency Quality Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| **Stability** | 7/10 | Fast-moving ML ecosystem requires active management |
| **Security** | 8.5/10 | No CVEs, actively maintained packages |
| **Maintenance** | 9/10 | All core packages well-maintained |
| **Platform Support** | 9/10 | NVIDIA, Intel, AMD coverage |
| **Flexibility** | 9.5/10 | Zero mandatory deps gives users control |
| **Overall** | **8.5/10** | Excellent dependency management |

---

## Potential Issues

### Heavyweight Dependencies
| Package | Size | Lighter Alternative |
|---------|------|---------------------|
| transformers | ~500MB | N/A (core requirement) |
| torch | 2GB+ | N/A (core requirement) |
| xformers | ~100MB | flash-attn (smaller) |

### Update Frequency Concerns
- **transformers**: Weekly releases, frequent breaking changes
- **trl**: Monthly releases, API changes
- **triton**: Quarterly, tied to NVIDIA drivers

---

## Recommended Installation

### For Development
```bash
pip install -e ".[huggingface,triton]"
```

### For Production (CUDA 12.4)
```bash
pip install unsloth[cu124only]
```

### For Windows
```bash
pip install unsloth[windows]
```

---

*Continue to [Metrics Summary](./metrics-summary.md) for quantitative analysis.*
