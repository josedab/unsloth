Model Registry
==============

.. module:: unsloth.registry

The registry module provides a system for tracking and searching pre-quantized
models available in the Unsloth model hub.

Functions
---------

.. autofunction:: search_models

.. autofunction:: register_models

Classes
-------

.. autoclass:: ModelInfo
   :members:
   :undoc-members:

.. autoclass:: QuantType
   :members:
   :undoc-members:

Example Usage
-------------

Search for models::

    from unsloth.registry import search_models, QuantType

    # Find all Llama 3.2 models
    models = search_models(base_name="Llama", version="3.2")

    # Find 4-bit quantized models
    models = search_models(quant_types=[QuantType.BNB])

    # Search by pattern
    models = search_models(search_pattern="8B-Instruct")

    for model in models:
        print(f"{model.model_path}: {model.size}B")
