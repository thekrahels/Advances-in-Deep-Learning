from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        #raise NotImplementedError()
        # Course-specific CoT prompt for MAK5724
        messages = [
            {
                "role": "system",
                "content": (
                "You are MAK5724's course assistant for Advances in Deep Learning at UT. "
                "Work through unit conversions carefully using the appropriate conversion factor. "
                "Keep the reasoning clear and straightforward. "
                "Always finish with the final numerical result in exactly this format: "
                "<answer>number</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How many centimeters are in 2 meters?",
            },
            {
                "role": "assistant",
                "content": (
                    "There are 100 centimeters in 1 meter. "
                    "So, 2 meters * 100 = 200 centimeters. "
                    "<answer>200</answer>"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )




def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
