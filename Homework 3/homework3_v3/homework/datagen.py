import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 5, temperature: float = 0.6, checkpoint_every: int = 25,):
    #raise NotImplementedError()
    """
    Generate a coverage-focused RFT dataset.

    Processes every training question and keeps at most one valid
    reasoning response per question.
    """

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    trainset = Dataset("train")
    generated_data = []

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for idx in range(len(trainset)):
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

        # Keep at most ONE valid generation for this question.
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

                # We found a valid response for this question.
                # Do not keep additional variants.
                break

        # Periodically save progress.
        if (idx + 1) % checkpoint_every == 0:
            with output_path.open("w") as f:
                json.dump(generated_data, f, indent=2)

            print(
                f"CHECKPOINT: processed {idx + 1} questions, "
                f"saved {len(generated_data)} unique examples",
                flush=True,
            )

    # Make sure the final state is always written.
    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Processed all {len(trainset)} training questions")
    print(f"Generated {len(generated_data)} accepted unique examples")
    print(f"Saved coverage dataset to {output_path}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
