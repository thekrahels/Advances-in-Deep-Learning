import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    #raise NotImplementedError()
    """
    Generate rejection fine-tuning data.

    For each training question:
    - generate multiple Chain-of-Thought responses
    - parse the numerical answer
    - keep only responses with a valid answer
    - save the accepted examples to JSON
    """

    # Use the larger SmolLM2 model to generate higher-quality reasoning.
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    trainset = Dataset("train")
    generated_data = []

    for idx in range(len(trainset)):
        question, correct_answer = trainset[idx]

        prompt = model.format_prompt(question)

        # Generate multiple diverse attempts for the same question.
        generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        for generation in generations:
            parsed_answer = model.parse_answer(generation)

            # Keep only generations that contain a valid numerical answer.
            if parsed_answer == parsed_answer and is_answer_valid(
                parsed_answer,
                correct_answer,
            ):
                generated_data.append(
                    [
                        question,
                        correct_answer,
                        generation,
                    ]
                )

        if (idx + 1) % 25 == 0:
            print(
                f"Processed {idx + 1}/{len(trainset)} questions - "
                f"accepted {len(generated_data)} examples"
            )

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)

    print(f"Generated {len(generated_data)} accepted examples")
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
