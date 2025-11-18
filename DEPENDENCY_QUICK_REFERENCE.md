# Unsloth Dependencies - Quick Reference

## Core At-A-Glance

| Dependency | Version | Purpose | Status |
|-----------|---------|---------|--------|
| **transformers** | >=4.51.3,<=4.57.2* | Model loading | Critical |
| **bitsandbytes** | >=0.45.5* | 4/8-bit quantization | Optional |
| **accelerate** | >=0.34.1 | Distributed training | Optional |
| **peft** | >=0.7.1* | LoRA/QLoRA | Optional |
| **trl** | >=0.18.2,<=0.24.0* | RLHF training | Optional |
| **xformers** | 0.0.22-0.0.31* | Attention optimization | Optional |
| **flash-attn** | >=2.6.3 | Fast attention (Linux) | Optional |
| **triton** | >=3.0.0 | Custom CUDA kernels | Optional |

*with selective version exclusions for known issues

---

## Version Exclusion Summary

```
transformers:   Excludes 4.52.0-3, 4.53.0, 4.54.0, 4.55.0-1, 4.57.0
bitsandbytes:   Excludes 0.46.0, 0.48.0
trl:            Excludes 0.19.0
peft:           Excludes 0.11.0
datasets:       Excludes 4.0.*, 4.1.0 (max <4.4.0)
```

**Why?** Known incompatibilities and bugs in these specific releases

---

## Installation Quick Commands

```bash
# Standard NVIDIA CUDA 12.1
pip install unsloth[cu121]

# NVIDIA CUDA 12.1 + Ampere GPU optimization
pip install unsloth[cu121-ampere]

# NVIDIA CUDA 12.4
pip install unsloth[cu124]

# Google Colab (optimized)
pip install unsloth[colab-new]

# Intel GPU (experimental)
pip install unsloth[intel-gpu-torch280]

# AMD GPU (experimental)
pip install unsloth[amd]

# Development mode
pip install -e ".[huggingface]"
```

---

## Build System

```
setuptools == 80.9.0
setuptools-scm == 9.2.0
```

---

## Platform Support

| Platform | Triton | Flash-Attn | XFormers | Notes |
|----------|--------|-----------|----------|-------|
| **Linux** | ✓ native | ✓ | ✓ | Full support |
| **Windows** | ✓ alternative | ✗ | ✓ cu124+ | Limited |
| **Intel GPU** | ✓ experimental | ✗ | ✗ | XPU backend |
| **AMD GPU** | ✗ | ✗ | ✗ | Custom bitsandbytes |

---

## Key Statistics

- **Python Support**: 3.9 - 3.13
- **PyTorch Versions**: 2.1.1 - 2.9.0 (with separate CUDA variants)
- **Installation Profiles**: 50+
- **XFormers Variants**: 70+
- **Unique Packages**: ~35
- **Excluded Versions**: 11+

---

## Dependency Health

| Aspect | Rating | Notes |
|--------|--------|-------|
| Stability | 7/10 | Dependencies update frequently |
| Security | 8.5/10 | No critical CVEs, actively maintained |
| Maintenance | 9/10 | All core libraries well-maintained |
| Platform Support | 9/10 | NVIDIA, Intel, AMD coverage |
| Flexibility | 9.5/10 | Users control PyTorch installation |

---

## Critical Decisions

1. **PyTorch NOT Required** 
   - Users install torch separately
   - Allows CUDA version flexibility
   
2. **Strategic Exclusions**
   - Bugs in transformers 4.52-4.55, 4.57
   - Bugs in bitsandbytes 0.46, 0.48
   - Breaking changes in trl 0.25+
   - API changes in datasets 4.0+

3. **50+ Pre-configured Profiles**
   - Different CUDA versions (11.8 - 12.8)
   - Different PyTorch versions (2.1.1 - 2.9.0)
   - Hardware-specific (Ampere, Intel GPU, etc.)
   - Platform-specific (Colab, Kaggle)

4. **70+ XFormers Variants**
   - Direct wheel URLs from PyTorch
   - Matched to PyTorch/CUDA combinations
   - Prevents version conflicts

---

## Dependency Tree (Top Level)

```
unsloth
├─ transformers (model loading)
├─ bitsandbytes (quantization)
├─ accelerate (distributed training)
├─ peft (parameter-efficient fine-tuning)
├─ trl (RLHF)
├─ xformers (attention optimization)
├─ flash-attn (fast attention)
├─ triton (custom CUDA kernels)
├─ datasets (dataset loading)
├─ huggingface_hub (model downloads)
└─ [utilities: numpy, tqdm, psutil, etc.]
```

---

## Known Issues & Workarounds

| Issue | Workaround |
|-------|-----------|
| transformers 4.52-4.55 incompatible | Use 4.51.3 or 4.57.2+ |
| bitsandbytes 0.46 crashes | Use 0.45.5 or 0.47+ |
| trl 0.25+ breaking changes | Pin to <=0.24.0 |
| datasets v4.x API changes | Use <4.4.0 |
| Flash-attn not on Windows | Use xformers instead |

---

## Installation Profiles Cheat Sheet

### By CUDA Version
- `cu118`: CUDA 11.8
- `cu121`: CUDA 12.1 (most common)
- `cu124`: CUDA 12.4
- `cu126`: CUDA 12.6
- `cu128`: CUDA 12.8

### By GPU Architecture
- Base profiles include standard xformers
- Add `-ampere` for Ampere GPUs (A100, RTX 30/40 series) + flash-attn
- `intel-gpu-torch*` for Intel Arc/Data Center
- `amd` for AMD Instinct/ROCm

### By PyTorch Version
- `torch211`: PyTorch 2.1.1
- `torch220`: PyTorch 2.2.0
- `torch230`: PyTorch 2.3.0
- `torch240`: PyTorch 2.4.0
- `torch250`: PyTorch 2.5.0
- `torch260`: PyTorch 2.6.0
- `torch270`: PyTorch 2.7.0
- `torch280`: PyTorch 2.8.0
- `torch290`: PyTorch 2.9.0

### Combination Examples
```
unsloth[cu121-torch250]           # CUDA 12.1 + PyTorch 2.5.0
unsloth[cu128-torch280]           # CUDA 12.8 + PyTorch 2.8.0
unsloth[cu121-ampere-torch250]    # + Ampere optimizations
```

---

## Next Steps if Dependencies Conflict

1. Identify conflicting package: `pip check`
2. Check your CUDA version: `nvidia-smi`
3. Use matching installation profile
4. If still issues, check version exclusion list
5. Avoid mixing multiple extra groups

---

Generated: Unsloth Dependency Analysis
File Location: /home/user/unsloth/DEPENDENCY_ANALYSIS.md
