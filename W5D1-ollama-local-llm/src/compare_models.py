from pathlib import Path

from src.ollama_client import chat_with_ollama


MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]

QUESTIONS = [
    "What is machine learning? Explain it for a beginner.",
    "What is overfitting and how can it be reduced?",
    "Explain precision and recall with a simple example.",
]

SYSTEM_PROMPT = """
You are a helpful AI and Machine Learning mentor.
Explain concepts clearly using beginner-friendly language,
short examples, and technically accurate information.
"""


def compare_models() -> str:
    """Run the same questions through both local models."""

    output = [
        "# Llama 3.2 vs Qwen 2.5 Comparison\n",
        "Both models were tested using the same system prompt "
        "and the same three questions.\n",
    ]

    for question_number, question in enumerate(QUESTIONS, start=1):

        output.append(f"## Question {question_number}\n")
        output.append(f"**Question:** {question}\n")

        for model in MODELS:

            print("=" * 70)
            print(f"Model: {model}")
            print(f"Question: {question}")

            try:
                response = chat_with_ollama(
                    prompt=question,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                )
            except RuntimeError as exc:
                response = f"ERROR: {exc}"

            print("\nResponse:")
            print(response)
            print()

            output.append(f"### {model}\n")
            output.append(f"{response}\n")

        output.append("---\n")

    return "\n".join(output)


if __name__ == "__main__":

    results = compare_models()

    output_path = Path("outputs/model_comparison.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        results,
        encoding="utf-8",
    )

    print(f"\nComparison saved to {output_path}")