import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import create_query_engine, ask_question


QUESTIONS = [
    "What is Retrieval Augmented Generation and why is it useful?",
    "Explain the difference between BM25 retrieval and dense retrieval.",
    "What is quantisation in LLMs and why is it useful for local inference?"
]


MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


def main():

    for model in MODELS:

        print("\n" + "#" * 70)
        print(f"MODEL: {model}")
        print("#" * 70)

        query_engine = create_query_engine(
            model_name=model
        )

        for question in QUESTIONS:

            print("\n" + "=" * 70)
            print(f"QUESTION: {question}")

            answer = ask_question(
                query_engine,
                question
            )

            print(f"ANSWER: {answer}")


if __name__ == "__main__":
    main()
