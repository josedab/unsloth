Quick Start Guide
=================

This guide will help you get started with Unsloth for finetuning language models.

Installation
------------

Install Unsloth with pip::

    pip install unsloth

For the latest features::

    pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"


Loading a Model
---------------

Load a pre-trained model with Unsloth optimizations::

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )


Adding LoRA Adapters
--------------------

Add LoRA adapters for efficient finetuning::

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


Training
--------

Train with the Unsloth trainer::

    from unsloth import UnslothTrainer, UnslothTrainingArguments

    trainer = UnslothTrainer(
        model=model,
        args=UnslothTrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


Saving the Model
----------------

Save as LoRA adapters::

    model.save_pretrained("./lora_model")

Save merged model::

    model.save_pretrained_merged(
        "merged_model",
        tokenizer,
        save_method="merged_16bit",
    )

Export to GGUF for llama.cpp::

    model.save_pretrained_gguf(
        "model",
        tokenizer,
        quantization_method="q4_k_m",
    )


Next Steps
----------

- See :doc:`api/loader` for full loading options
- See :doc:`api/trainer` for training configuration
- See :doc:`api/save` for export options
