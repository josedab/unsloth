Unsloth Documentation
=====================

Unsloth makes finetuning large language models 2x faster and uses 70% less memory.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/loader
   api/trainer
   api/save
   api/registry

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/finetuning
   guides/inference
   guides/export

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Quick Start
-----------

Install Unsloth::

    pip install unsloth

Load and finetune a model::

    from unsloth import FastLanguageModel

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Train with your data
    from unsloth import UnslothTrainer, UnslothTrainingArguments

    trainer = UnslothTrainer(
        model=model,
        args=UnslothTrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()


Features
--------

- **2x Faster Training**: Optimized Triton kernels and Flash Attention
- **70% Less Memory**: Efficient gradient checkpointing and memory management
- **Easy to Use**: Drop-in replacement for HuggingFace workflows
- **Multiple Formats**: Export to GGUF, merged weights, or LoRA adapters
- **Wide Compatibility**: Supports Llama, Mistral, Gemma, Qwen, and more


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
