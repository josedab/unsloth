Contributing to Documentation
=============================

This guide explains how to contribute to Unsloth's documentation, including
docstring standards, building docs locally, and best practices.


Docstring Standard
------------------

All docstrings in Unsloth follow the **Google Style** format for consistency
and compatibility with Sphinx Napoleon extension.


Required Sections
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Section
     - Required
     - Description
   * - Summary
     - Yes
     - One-line description of the function
   * - Description
     - For complex funcs
     - Extended explanation
   * - Args
     - Yes
     - All parameters with types and descriptions
   * - Returns
     - Yes
     - Return value(s) with type
   * - Raises
     - If applicable
     - Exceptions that may be raised
   * - Example
     - Yes for public API
     - Runnable code example
   * - Note
     - If applicable
     - Important caveats or warnings
   * - See Also
     - If applicable
     - Related functions or classes


Example Docstring
^^^^^^^^^^^^^^^^^

Here's a complete example following our standard::

    def load_model(
        model_name: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a model with Unsloth optimizations.

        This function loads a HuggingFace model and applies Unsloth
        optimizations including Flash Attention and memory-efficient
        gradient checkpointing.

        Args:
            model_name: HuggingFace model ID (e.g., "unsloth/llama-3-8b")
                or local path to model directory.
            max_seq_length: Maximum sequence length for training. If None,
                uses the model's default from config.json.
            load_in_4bit: Whether to load model in 4-bit quantization.
                Reduces memory by ~75% with minimal quality loss.

        Returns:
            Tuple of (model, tokenizer) ready for training or inference.

        Raises:
            ValueError: If model_name is not a supported architecture.
            RuntimeError: If model loading fails due to memory issues.

        Example:
            Basic usage::

                >>> model, tokenizer = load_model(
                ...     "unsloth/llama-3-8b",
                ...     max_seq_length=2048,
                ... )

        Note:
            Import unsloth before transformers to ensure patches are applied.

        See Also:
            - :meth:`get_peft_model`: Apply LoRA after loading
        """


Building Documentation
----------------------

Prerequisites
^^^^^^^^^^^^^

Install the documentation dependencies::

    pip install sphinx sphinx-rtd-theme

Building Locally
^^^^^^^^^^^^^^^^

From the ``docs/`` directory::

    # Build HTML documentation
    make html

    # View in browser
    open _build/html/index.html

The built documentation will be in ``docs/_build/html/``.


Adding New Documentation
------------------------

For New Functions
^^^^^^^^^^^^^^^^^

1. Add a Google-style docstring to your function
2. Include all required sections (Args, Returns, Example)
3. Add the function to the appropriate ``.rst`` file in ``docs/api/``
4. Rebuild docs and verify rendering

For New Modules
^^^^^^^^^^^^^^^

1. Create a new ``.rst`` file in ``docs/api/``
2. Add autodoc directives for the module
3. Add the file to the toctree in ``docs/index.rst``
4. Add module docstring to the Python file


Best Practices
--------------

1. **Be Specific**: Include valid values, ranges, and defaults
2. **Show Examples**: Every public function should have a runnable example
3. **Cross-Reference**: Use ``:func:``, ``:class:``, ``:meth:`` to link related items
4. **Stay Updated**: Update docstrings when changing function behavior
5. **Test Examples**: Ensure code examples actually work


Verifying Documentation
-----------------------

Run doctest to verify examples::

    python -m doctest unsloth/models/loader.py -v

Check for Sphinx warnings::

    make html 2>&1 | grep -i warning


Priority Areas
--------------

Documentation contributions are especially welcome for:

1. **Public API** (highest priority)
   - ``models/loader.py``
   - ``save.py``
   - ``trainer.py``

2. **Core Internals**
   - ``models/llama.py``
   - ``models/_utils.py``
   - ``tokenizer_utils.py``

3. **Kernels** (lower priority, for advanced users)
   - ``kernels/fast_lora.py``
   - ``kernels/utils.py``


Questions?
----------

If you have questions about documentation standards or need help, please
open an issue on GitHub: https://github.com/unslothai/unsloth/issues
