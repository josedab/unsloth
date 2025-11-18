"""
Model registry for Unsloth pre-quantized models.

This module provides a registry system for tracking and searching pre-quantized
models available in the Unsloth model hub. It includes information about model
architectures, sizes, quantization types, and metadata.

The primary components are:

- :func:`search_models`: Search for models by various criteria
- :func:`register_models`: Initialize the model registry
- :class:`ModelInfo`: Data class containing model metadata
- :class:`QuantType`: Enum of quantization types (BNB, UNSLOTH, GGUF, etc.)

Example:
    Search for Llama models::

        from unsloth.registry import search_models, QuantType

        # Find all Llama 3.2 models
        models = search_models(base_name="Llama", version="3.2")

        # Find 4-bit quantized models
        models = search_models(quant_types=[QuantType.BNB])

        # Search by pattern
        models = search_models(search_pattern="8B-Instruct")

        for model in models:
            print(f"{model.model_path}: {model.size}B")

See Also:
    - :mod:`unsloth.models.loader`: For loading registered models
"""

from ._deepseek import register_deepseek_models as _register_deepseek_models
from ._gemma import register_gemma_models as _register_gemma_models
from ._llama import register_llama_models as _register_llama_models
from ._mistral import register_mistral_models as _register_mistral_models
from ._phi import register_phi_models as _register_phi_models
from ._qwen import register_qwen_models as _register_qwen_models
from .registry import MODEL_REGISTRY, ModelInfo, QuantType

_ARE_MODELS_REGISTERED = False


def register_models():
    """
    Initialize the model registry with all supported models.

    This function populates the MODEL_REGISTRY with information about all
    pre-quantized models available in the Unsloth model hub. It registers
    models for all supported architectures including Llama, Mistral, Gemma,
    Qwen, Phi, and DeepSeek.

    This function is automatically called by :func:`search_models` if the
    registry has not been initialized. You typically don't need to call
    this directly.

    Example:
        >>> from unsloth.registry import register_models, MODEL_REGISTRY
        >>> register_models()
        >>> print(f"Registered {len(MODEL_REGISTRY)} models")

    Note:
        Safe to call multiple times - subsequent calls are no-ops.
    """
    global _ARE_MODELS_REGISTERED

    if _ARE_MODELS_REGISTERED:
        return
    _register_deepseek_models()
    _register_gemma_models()
    _register_llama_models()
    _register_mistral_models()
    _register_phi_models()
    _register_qwen_models()

    _ARE_MODELS_REGISTERED = True


def search_models(
    org: str = None,
    base_name: str = None,
    version: str = None,
    size: str = None,
    quant_types: list[QuantType] = None,
    search_pattern: str = None,
) -> list[ModelInfo]:
    """
    Search for models in the registry by various criteria.

    This function allows filtering the model registry by organization,
    base name, version, size, quantization type, or a general search pattern.
    All filters are applied with AND logic.

    Args:
        org: Filter by organization (e.g., "unsloth", "meta-llama").
        base_name: Filter by base model name (e.g., "Llama", "Mistral").
        version: Filter by model version (e.g., "3.2", "2").
        size: Filter by model size (e.g., "8", "70").
        quant_types: Filter by quantization types. Pass a list of QuantType
            values to include models with any of those quantization types.
        search_pattern: Search pattern to match against the full model path
            (model_id on HuggingFace Hub).

    Returns:
        list[ModelInfo]: List of ModelInfo objects matching the search criteria.

    Example:
        Find all Llama 3.2 models::

            >>> models = search_models(base_name="Llama", version="3.2")
            >>> for m in models[:3]:
            ...     print(m.model_path)
            unsloth/Llama-3.2-1B-Instruct-bnb-4bit
            unsloth/Llama-3.2-3B-Instruct-bnb-4bit
            ...

        Find 8B instruction-tuned models::

            >>> models = search_models(size="8", search_pattern="Instruct")

        Find all BNB quantized models::

            >>> from unsloth.registry import QuantType
            >>> models = search_models(quant_types=[QuantType.BNB])

    See Also:
        - :class:`ModelInfo`: Model metadata structure
        - :class:`QuantType`: Available quantization types
    """
    if not _ARE_MODELS_REGISTERED:
        register_models()

    model_infos = MODEL_REGISTRY.values()
    if org:
        model_infos = [
            model_info for model_info in model_infos if model_info.org == org
        ]
    if base_name:
        model_infos = [
            model_info
            for model_info in model_infos
            if model_info.base_name == base_name
        ]
    if version:
        model_infos = [
            model_info for model_info in model_infos if model_info.version == version
        ]
    if size:
        model_infos = [
            model_info for model_info in model_infos if model_info.size == size
        ]
    if quant_types:
        model_infos = [
            model_info
            for model_info in model_infos
            if any(model_info.quant_type == quant_type for quant_type in quant_types)
        ]
    if search_pattern:
        model_infos = [
            model_info
            for model_info in model_infos
            if search_pattern in model_info.model_path
        ]

    return model_infos
