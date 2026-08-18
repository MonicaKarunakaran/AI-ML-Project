import sys
from pathlib import Path

# Add project root to Python path so "src" can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chain import ask_question


QUESTIONS = [
    "What is machine learning?",
    "What is the difference between AI and machine learning?",
    "What is a neural network?",
    "What is overfitting?",
    "What is LangChain?",
]


def main():
    print("=" * 60)
    print("W6D2 - LANGCHAIN CHAIN DEMO")
    print("=" * 60)

    results = []

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n--- Input {index} ---")
        print(f"Question: {question}")

        try:
            answer = ask_question(question)

            print(f"Answer: {answer}")

            results.append(
                f"--- Input {index} ---\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n"
            )

        except Exception as exc:
            print(f"Error: {exc}")

            results.append(
                f"--- Input {index} ---\n"
                f"Question: {question}\n"
                f"Error: {exc}\n"
            )

    output_file = PROJECT_ROOT / "outputs" / "chain_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(results))

    print("\n" + "=" * 60)
    print("Chain execution completed.")
    print(f"Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()