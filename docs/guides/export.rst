Export Guide
============

This guide covers exporting models to various formats.

.. note::
   This guide is under construction. Please check back later for complete content.


Export Formats
--------------

Unsloth supports exporting to:

- LoRA adapters
- Merged 16-bit weights
- Merged 4-bit weights
- GGUF for llama.cpp


GGUF Export
-----------

Export to GGUF for use with llama.cpp, Ollama, or LM Studio::

    model.save_pretrained_gguf(
        "model",
        tokenizer,
        quantization_method="q4_k_m",
    )

See :func:`unsloth.save.print_quantization_methods` for all available
quantization options.


Push to HuggingFace Hub
-----------------------

Push your model directly to HuggingFace::

    model.push_to_hub_merged(
        "username/model-name",
        tokenizer,
        save_method="merged_16bit",
        token="hf_xxxxx",
    )
