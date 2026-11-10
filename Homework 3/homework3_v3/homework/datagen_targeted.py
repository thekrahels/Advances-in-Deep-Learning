import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid


TARGET_PAIRS = [
    ("kb", "bit"),
    ("week", "day"),
    ("week", "h"),
    ("week", "month"),
    ("month", "h"),
    ("decade", "month"),
    ("century", "month"),
    ("ft", "yd"),
    ("ft", " m"),
    ("yd", "mm"),
    ("mile", "ft"),
    ("kmh", "ft/s"),
    ("km/h", "ft/s"),
    ("kmh", "in/s"),
    ("km/h", "in/s"),
    ("m/s", "in/s"),
    ("litre", "gallon"),
    ("liter", "gallon"),
    ("quart", "gallon"),
    ("quart", "litre"),
    ("quart", "liter"),
    ("pound", "kg"),
    ("cm", " in"),
]


def generate_targeted(
    output_json: str = "data/rft_targeted.json",
    oversample: int = 5,
    temperature: float = 0.6,
):
    trainset = Dataset("train")
    model = CoTModel(
        checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    )

    original_path = Path("data/rft.json")

    with original_path.open() as f:
        original = json.load(f)

    existing_questions = {x[0] for x in original}

    candidates = []

    for idx in range(len(trainset)):
        question, answer = trainset[idx]
        q_lower = question.lower()

        if question in existing_questions:
            continue

        if any(a in q_lower and b in q_lower for a, b in TARGET_PAIRS):
            candidates.append((question, answer))

    print(
        f"Found {len(candidates)} targeted training questions "
        f"not already represented in rft.json",
        flush=True,
    )

    accepted = []

    for idx, (question, correct_answer) in enumerate(candidates):
        print(
            f"Targeted question {idx + 1}/{len(candidates)} "
            f"(accepted: {len(accepted)})",
            flush=True,
        )

        prompt = model.format_prompt(question)

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
                accepted.append(
                    [question, correct_answer, generation]
                )
                break

        # Save regularly in case generation is interrupted.
        if (idx + 1) % 25 == 0:
            with Path(output_json).open("w") as f:
                json.dump(accepted, f, indent=2)

            print(
                f"CHECKPOINT: {len(accepted)} accepted",
                flush=True,
            )

    with Path(output_json).open("w") as f:
        json.dump(accepted, f, indent=2)

    print(f"Targeted accepted examples: {len(accepted)}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_targeted)