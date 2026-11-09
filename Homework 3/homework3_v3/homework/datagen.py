import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6, target_examples: int = 1000, checkpoint_every: int = 25,):
    #raise NotImplementedError()
    """
    Generate rejection fine-tuning data.

    For each training question:
    - generate multiple Chain-of-Thought responses
    - parse the numerical answer
    - keep only responses with a valid answer
    - periodically checkpoint the accepted data
    - stop once target_examples valid examples have been collected
    """

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    trainset = Dataset("train")
    generated_data = []

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx in range(len(trainset)):
        question, correct_answer = trainset[idx]

        prompt = model.format_prompt(question)

        print(
            f"Generating question {idx + 1}/{len(trainset)} "
            f"(accepted so far: {len(generated_data)})",
            flush=True,
        )

        generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

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

                # Stop accepting once the target is reached.
                if len(generated_data) >= target_examples:
                    break

        # Save a checkpoint periodically.
        if (idx + 1) % checkpoint_every == 0:
            with output_path.open("w") as f:
                json.dump(generated_data, f, indent=2)

            print(
                f"CHECKPOINT: saved {len(generated_data)} examples "
                f"after question {idx + 1}",
                flush=True,
            )

        # Stop the entire generation loop at the target.
        if len(generated_data) >= target_examples:
            print(
                f"Target of {target_examples} accepted examples reached.",
                flush=True,
            )
            break

    # Always save once more at the end.
    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Generated {len(generated_data)} accepted examples")
    print(f"Saved final dataset to {output_path}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
