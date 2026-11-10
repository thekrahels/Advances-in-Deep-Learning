from .base_llm import BaseLLM
from .data import Dataset
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

def format_example(
    prompt: str,
    correct_answer: float,
    reasoning: str,
) -> dict[str, str]:
    """
    Format an accepted RFT example for training.
    The correct_answer was used during rejection sampling to verify the
    generated reasoning, so training uses the accepted reasoning itself.
    """
    return {
        "question": prompt,
        "answer": reasoning,
    }

def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #raise NotImplementedError()
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    
    # Start RFT from the original base model.
    llm = BaseLLM()

    # Configure a new LoRA adapter for RFT.
    lora_config = LoraConfig(r=8,
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
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=4e-4,
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
