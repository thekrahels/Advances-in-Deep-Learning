import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 5, temperature: float = 0.6, checkpoint_every: int = 25, start_question: int = 1,):
    #raise NotImplementedError()
    """
    Generate a coverage-focused RFT dataset.

    Keeps at most one valid reasoning response per question.
    If output_json already exists, previously accepted examples are loaded
    so generation can resume without losing checkpointed work.
    """

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    trainset = Dataset("train")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume with already-saved accepted examples when available.
    if output_path.exists():
        with output_path.open("r") as f:
            generated_data = json.load(f)

        print(
            f"RESUME: loaded {len(generated_data)} previously accepted examples",
            flush=True,
        )
    else:
        generated_data = []

    # start_question is human-readable: 1 means dataset index 0.
    start_idx = start_question - 1

    for idx in range(start_idx, len(trainset)):
        question, correct_answer = trainset[idx]

        print(
            f"Generating question {idx + 1}/{len(trainset)} "
            f"(accepted so far: {len(generated_data)})",
            flush=True,
        )

        prompt = model.format_prompt(question)

        generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        # Keep at most one valid generation for this question.
        for generation in generations:
            parsed_answer = model.parse_answer(generation)

            if (
                parsed_answer == parsed_answer
                and is_answer_valid(parsed_answer, correct_answer)
            ):
                generated_data.append(
                    [
                        question,
                        correct_answer,
                        generation,
                    ]
                )
                break

        # Checkpoint based on the actual training-question number.
        if (idx + 1) % checkpoint_every == 0:
            with output_path.open("w") as f:
                json.dump(generated_data, f, indent=2)

            print(
                f"CHECKPOINT: processed {idx + 1} questions, "
                f"saved {len(generated_data)} unique examples",
                flush=True,
            )

    # Final save.
    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Processed through question {len(trainset)}")
    print(f"Generated {len(generated_data)} accepted unique examples")
    print(f"Saved coverage dataset to {output_path}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
