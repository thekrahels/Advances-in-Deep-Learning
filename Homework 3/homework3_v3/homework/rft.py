from .base_llm import BaseLLM
from .sft import test_model
from .sft import TokenizedDataset, test_model

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
    # Load the base 360M SmolLM2 model used by the homework grader.
    llm = BaseLLM()

    # Configure the LoRA adapter.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)

    # Reduce GPU memory usage during training.
    llm.model.enable_input_require_grads()
    llm.model.gradient_checkpointing_enable()

    # Load the accepted reasoning examples from data/rft.json.
    train_data = Dataset("rft")
    tokenized_data = TokenizedDataset(
        llm.tokenizer,
        train_data,
        format_example,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="tensorboard",
        **kwargs,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()

    # Save the final LoRA adapter directly where load() expects it.
    trainer.model.save_pretrained(output_dir)

    # Evaluate the completed RFT adapter.
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
