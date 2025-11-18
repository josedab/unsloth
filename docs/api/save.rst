Saving and Export
=================

.. module:: unsloth.save

The save module provides functions for saving trained models in various formats.

Functions
---------

.. autofunction:: unsloth_save_model

.. autofunction:: print_quantization_methods

.. autofunction:: create_huggingface_repo

Quantization Methods
--------------------

The following quantization methods are available for GGUF export:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``q4_k_m``
     - Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2 tensors, else Q4_K
   * - ``q5_k_m``
     - Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2 tensors, else Q5_K
   * - ``q8_0``
     - Fast conversion. High resource use, but generally acceptable
   * - ``f16``
     - Float16 - Fastest conversion + retains 100% accuracy
   * - ``bf16``
     - Bfloat16 - Fastest conversion + retains 100% accuracy

Use :func:`print_quantization_methods` to see all available methods.
