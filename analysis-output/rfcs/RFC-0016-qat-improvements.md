# RFC-0016: Quantization-Aware Training (QAT) Improvements

**Status:** Draft
**Author:** Codebase Analysis
**Created:** November 18, 2025
**Effort:** 14 person-days
**Priority:** P2 (Long-term)

---

## Summary

Enhance Quantization-Aware Training (QAT) support to produce higher-quality quantized models with better accuracy retention, lower perplexity, and optimized inference performance compared to post-training quantization.

---

## Motivation

Unsloth currently supports post-training quantization (PTQ) via bitsandbytes, but lacks proper QAT implementation. QAT can significantly improve quantized model quality:

### PTQ vs QAT Quality Comparison

| Metric | Full Precision | PTQ (4-bit) | QAT (4-bit) |
|--------|---------------|-------------|-------------|
| Perplexity | 5.2 | 5.8 (+11%) | 5.3 (+2%) |
| Accuracy | 100% | 92% | 98% |
| Inference Speed | 1.0x | 3.5x | 3.5x |

**Key Insight**: QAT provides similar speedup to PTQ but with much better accuracy retention.

### Current Limitations

```python
# Current approach - post-training quantization only
model, tokenizer = FastLanguageModel.from_pretrained(
    "model-name",
    load_in_4bit=True,  # Applied after training, quality loss
)

# No QAT support
# Cannot train with quantization in the loop
# Cannot fine-tune quantization parameters
```

### Impact

| Issue | Effect |
|-------|--------|
| No QAT support | Lower quantized model quality |
| PTQ only | Higher perplexity after quantization |
| No mixed precision QAT | Cannot optimize quantization per layer |
| No calibration dataset | Suboptimal quantization parameters |

---

## Detailed Design

### 1. QAT Training Wrapper

```python
# unsloth/qat/trainer.py
import torch
import torch.nn as nn
from typing import Optional, Literal
from torch.quantization import (
    prepare_qat,
    convert,
    QConfig,
    default_qat_qconfig,
)

class QATConfig:
    """
    Configuration for Quantization-Aware Training.

    Args:
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        weight_bits: Bits for weights (4, 8)
        activation_bits: Bits for activations (8, 16)
        symmetric: Use symmetric quantization
        per_channel: Per-channel quantization (vs per-tensor)
        calibration_steps: Steps to calibrate quantization ranges

    Example:
        >>> config = QATConfig(
        ...     weight_bits=4,
        ...     activation_bits=8,
        ...     per_channel=True,
        ... )
    """

    def __init__(
        self,
        backend: Literal["fbgemm", "qnnpack"] = "fbgemm",
        weight_bits: Literal[4, 8] = 4,
        activation_bits: Literal[8, 16] = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        calibration_steps: int = 100,
    ):
        self.backend = backend
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.calibration_steps = calibration_steps

class QATWrapper(nn.Module):
    """
    Wrap a model for Quantization-Aware Training.

    Adds fake quantization operations that simulate quantization
    during training, allowing the model to learn quantization-friendly
    representations.

    Usage:
        >>> from unsloth import FastLanguageModel
        >>> from unsloth.qat import QATWrapper, QATConfig
        >>>
        >>> # Load model
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>>
        >>> # Wrap for QAT
        >>> qat_config = QATConfig(weight_bits=4)
        >>> model = QATWrapper(model, qat_config)
        >>>
        >>> # Train as normal
        >>> trainer = UnslothTrainer(model=model, ...)
        >>> trainer.train()
        >>>
        >>> # Convert to actual quantized model
        >>> quantized_model = model.convert_to_quantized()
    """

    def __init__(
        self,
        model: nn.Module,
        config: QATConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.qconfig = self._create_qconfig()

        # Prepare model for QAT
        self._prepare_qat()

    def _create_qconfig(self) -> QConfig:
        """Create quantization config based on settings."""
        if self.config.weight_bits == 4:
            # 4-bit quantization config
            from torch.ao.quantization import (
                FakeQuantize,
                MinMaxObserver,
            )

            weight_observer = MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric if self.config.per_channel
                    else torch.per_tensor_symmetric,
                reduce_range=True,  # For 4-bit
            )

            activation_observer = MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            )

            qconfig = QConfig(
                activation=FakeQuantize.with_args(
                    observer=activation_observer,
                    quant_min=0,
                    quant_max=255 if self.config.activation_bits == 8 else 65535,
                ),
                weight=FakeQuantize.with_args(
                    observer=weight_observer,
                    quant_min=-8 if self.config.weight_bits == 4 else -128,
                    quant_max=7 if self.config.weight_bits == 4 else 127,
                ),
            )

        else:
            # Use default 8-bit config
            qconfig = default_qat_qconfig

        return qconfig

    def _prepare_qat(self):
        """Prepare model for QAT."""
        # Set backend
        torch.backends.quantized.engine = self.config.backend

        # Apply qconfig to model
        self.model.qconfig = self.qconfig

        # Prepare for QAT (adds fake quant modules)
        prepare_qat(self.model, inplace=True)

    def forward(self, *args, **kwargs):
        """Forward pass with fake quantization."""
        return self.model(*args, **kwargs)

    def convert_to_quantized(self) -> nn.Module:
        """
        Convert QAT model to actual quantized model.

        Returns:
            Fully quantized model ready for inference
        """
        # Switch to eval mode
        self.model.eval()

        # Convert fake quant to actual quant
        quantized_model = convert(self.model.cpu(), inplace=False)

        return quantized_model

    def calibrate(self, calibration_dataloader):
        """
        Calibrate quantization ranges.

        Args:
            calibration_dataloader: DataLoader with calibration data
        """
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= self.config.calibration_steps:
                    break

                # Forward pass to collect statistics
                self.model(**batch)

        self.model.train()
```

### 2. Mixed-Precision QAT

```python
# unsloth/qat/mixed_precision.py
from typing import Dict, List
import torch.nn as nn

class MixedPrecisionQAT:
    """
    Apply different quantization levels to different layers.

    Strategy:
    - Embedding/output layers: 8-bit (sensitive to quantization)
    - Attention layers: 8-bit (sensitive)
    - FFN layers: 4-bit (robust to quantization)

    Example:
        >>> config = MixedPrecisionConfig({
        ...     "embed_tokens": 8,
        ...     "self_attn": 8,
        ...     "mlp": 4,
        ... })
        >>> model = MixedPrecisionQAT(model, config)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_configs: Dict[str, int],  # layer name pattern -> bits
    ):
        self.model = model
        self.layer_configs = layer_configs

        self._apply_mixed_precision()

    def _apply_mixed_precision(self):
        """Apply different quantization configs to different layers."""
        for name, module in self.model.named_modules():
            # Determine bit width for this layer
            bits = self._get_bits_for_layer(name)

            if bits:
                # Apply appropriate QConfig
                module.qconfig = self._create_qconfig(bits)

    def _get_bits_for_layer(self, layer_name: str) -> Optional[int]:
        """Get quantization bits for a layer based on name."""
        for pattern, bits in self.layer_configs.items():
            if pattern in layer_name:
                return bits
        return None

    def _create_qconfig(self, bits: int):
        """Create QConfig for specific bit width."""
        # Implementation similar to QATWrapper._create_qconfig
        pass
```

### 3. LoRA-QAT Integration

```python
# unsloth/qat/lora_qat.py
from peft import LoraConfig, get_peft_model

class LoRAQATModel(nn.Module):
    """
    Combine LoRA and QAT for efficient fine-tuning.

    Strategy:
    1. Apply LoRA adapters (trainable)
    2. Quantize base model weights (frozen, quantized)
    3. Train LoRA with quantization-aware loss

    Benefits:
    - Low memory (quantized base + small LoRA)
    - High quality (QAT-trained)
    - Fast training (only LoRA trainable)

    Example:
        >>> model = FastLanguageModel.from_pretrained(...)
        >>> model = LoRAQATModel(
        ...     model,
        ...     lora_config=LoraConfig(r=16),
        ...     qat_config=QATConfig(weight_bits=4),
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoraConfig,
        qat_config: QATConfig,
    ):
        super().__init__()

        # Apply LoRA
        self.model = get_peft_model(model, lora_config)

        # Apply QAT to base model only
        self._apply_qat_to_base(qat_config)

    def _apply_qat_to_base(self, qat_config: QATConfig):
        """Apply QAT only to base model, not LoRA adapters."""
        for name, module in self.model.named_modules():
            # Skip LoRA modules
            if "lora" in name.lower():
                continue

            # Apply QAT qconfig
            if hasattr(module, "weight"):
                module.qconfig = QATWrapper(None, qat_config).qconfig
```

### 4. QAT-Optimized Export

```python
# unsloth/qat/export.py
import torch

def export_qat_model(
    model: QATWrapper,
    output_path: str,
    format: Literal["onnx", "torchscript", "gguf"] = "onnx",
):
    """
    Export QAT model for optimized inference.

    Args:
        model: QAT-trained model
        output_path: Where to save exported model
        format: Export format

    Example:
        >>> export_qat_model(
        ...     model,
        ...     "model_int4.onnx",
        ...     format="onnx",
        ... )
    """
    # Convert to quantized
    quantized = model.convert_to_quantized()

    if format == "onnx":
        _export_onnx(quantized, output_path)
    elif format == "torchscript":
        _export_torchscript(quantized, output_path)
    elif format == "gguf":
        _export_gguf(quantized, output_path)

def _export_onnx(model, output_path):
    """Export to ONNX with quantization."""
    dummy_input = torch.randint(0, 1000, (1, 128))

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
    )
```

---

## Example Usage

### Basic QAT Training

```python
from unsloth import FastLanguageModel
from unsloth.qat import QATWrapper, QATConfig

# Load model (full precision)
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3-8B",
    max_seq_length=2048,
)

# Wrap for QAT
qat_config = QATConfig(
    weight_bits=4,
    activation_bits=8,
    per_channel=True,
)
model = QATWrapper(model, qat_config)

# Train as normal
trainer = UnslothTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Convert to quantized for inference
quantized_model = model.convert_to_quantized()
quantized_model.save_pretrained("model_qat_int4")
```

### LoRA + QAT

```python
from unsloth.qat import LoRAQATModel

# Combine LoRA and QAT
model = LoRAQATModel(
    model,
    lora_config=LoraConfig(r=16, lora_alpha=16),
    qat_config=QATConfig(weight_bits=4),
)

# Train
trainer.train()

# Export quantized model with LoRA merged
quantized = model.convert_to_quantized()
```

### Mixed-Precision QAT

```python
from unsloth.qat import MixedPrecisionQAT

# Different precision for different layers
model = MixedPrecisionQAT(
    model,
    layer_configs={
        "embed": 8,      # Embeddings: 8-bit
        "attn": 8,       # Attention: 8-bit
        "mlp": 4,        # FFN: 4-bit
        "lm_head": 8,    # Output: 8-bit
    },
)
```

---

## Implementation Plan

### Phase 1: Core QAT (Days 1-5)

**Days 1-2: QAT Infrastructure**
- Implement QATConfig
- Implement QATWrapper
- Add fake quantization

**Days 3-4: Training Integration**
- Integrate with UnslothTrainer
- Add calibration support
- Test training convergence

**Day 5: Conversion**
- Implement convert_to_quantized()
- Test inference quality
- Benchmark performance

### Phase 2: Advanced Features (Days 6-10)

**Days 6-7: Mixed Precision**
- Implement MixedPrecisionQAT
- Add layer-wise configs
- Test quality improvements

**Days 8-9: LoRA Integration**
- Implement LoRAQATModel
- Test memory usage
- Benchmark training speed

**Day 10: Export**
- Implement export functions
- Support ONNX, TorchScript, GGUF
- Test exported models

### Phase 3: Testing and Documentation (Days 11-14)

**Days 11-12: Testing**
- Test various bit widths
- Test mixed precision
- Compare quality vs PTQ

**Days 13-14: Documentation**
- Write QAT guide
- Add examples
- Create comparison benchmarks

### Milestones

| Day | Deliverable |
|-----|-------------|
| 5 | Basic QAT working |
| 10 | Advanced features complete |
| 14 | Full release ready |

---

## Backwards Compatibility

### Breaking Changes

None. QAT is opt-in.

### Migration

```python
# Old (PTQ)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
)

# New (QAT) - better quality
model, tokenizer = FastLanguageModel.from_pretrained(model_name)
model = QATWrapper(model, QATConfig(weight_bits=4))
trainer.train()
quantized = model.convert_to_quantized()
```

---

## Alternatives Considered

### Alternative 1: PTQ Only

Continue with post-training quantization only.

**Rejected:**
- Lower quality
- Higher perplexity
- Suboptimal for deployment

### Alternative 2: Third-Party QAT

Use PyTorch's quantization or NVIDIA's quantization toolkit.

**Rejected:**
- Not integrated with Unsloth workflow
- Missing optimizations
- Different API

### Alternative 3: GPTQ/AWQ

Use GPTQ or AWQ for quantization.

**Rejected:**
- Complementary, not alternative
- Can be added later
- QAT more flexible

---

## Open Questions

1. **Best default bit widths?**
   - 4-bit weights, 8-bit activations?
   - Test empirically

2. **Calibration dataset size?**
   - 100 steps sufficient?
   - Model-dependent?

3. **Support FP8 for H100?**
   - Hardware support available
   - Software support maturing

---

## Success Criteria

- [ ] QAT training works correctly
- [ ] Quantized models have <5% perplexity increase
- [ ] Mixed precision QAT improves quality
- [ ] LoRA + QAT reduces memory vs full fine-tuning
- [ ] Export to multiple formats works
- [ ] Documentation complete

### Measurable Outcomes

| Metric | PTQ | QAT |
|--------|-----|-----|
| Perplexity increase | 10-15% | <5% |
| Accuracy retention | 90-95% | 95-99% |
| Training time | N/A | 1.2x slower |
| Inference speed | 3.5x | 3.5x |

---

## Required Approvals

- [ ] Technical review
- [ ] Quality benchmarking
- [ ] Documentation review

---

## Rollback Strategy

1. QAT is opt-in feature
2. PTQ remains default
3. Can disable QAT without breaking changes

---

## References

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- QAT Tutorial: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
- NVIDIA Quantization Toolkit: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- GPTQ Paper: https://arxiv.org/abs/2210.17323

---

*Return to [Prioritization Matrix](./00-prioritization-matrix.md)*
