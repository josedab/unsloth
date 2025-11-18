Installation
============

This page covers installation options for Unsloth.


Requirements
------------

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)


Basic Installation
------------------

Install from PyPI::

    pip install unsloth


Development Installation
------------------------

Install from source for the latest features::

    pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"


Colab / Kaggle
--------------

For Colab or Kaggle notebooks, use::

    %%capture
    !pip install unsloth


Optional Dependencies
---------------------

For GGUF export (llama.cpp)::

    pip install llama-cpp-python

For fast inference with vLLM::

    pip install vllm

For Flash Attention (recommended)::

    pip install flash-attn --no-build-isolation


Verifying Installation
----------------------

Verify your installation::

    import unsloth
    from unsloth import FastLanguageModel

    print("Unsloth installed successfully!")


Troubleshooting
---------------

**CUDA not found**: Ensure you have CUDA installed and PyTorch with CUDA support.

**Out of memory**: Try reducing ``max_seq_length`` or enabling ``load_in_4bit=True``.

**Import errors**: Make sure to import unsloth before other libraries::

    import unsloth  # Must be first!
    from transformers import ...
