Inference Guide
===============

This guide covers using Unsloth models for inference.

.. note::
   This guide is under construction. Please check back later for complete content.


Basic Inference
---------------

Load a finetuned model for inference::

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "./your_finetuned_model",
        max_seq_length=2048,
    )

    # Enable fast inference mode
    FastLanguageModel.for_inference(model)


Fast Inference with vLLM
------------------------

For production inference, use the vLLM backend::

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        fast_inference=True,
        gpu_memory_utilization=0.8,
    )
