from src.chain import ask_question


def main():
    questions = [
        "What is LangChain?",
        "What is RAG?",
        "What is an LLM?",
        "What is prompt engineering?",
        "What is an AI agent?",
    ]

    print("=" * 60)
    print("W6D3 - LANGCHAIN CHAIN")
    print("=" * 60)

    for i, question in enumerate(questions, start=1):
        print(f"\nInput {i}: {question}")
        answer = ask_question(question)
        print(f"Output: {answer}")


if __name__ == "__main__":
    main()