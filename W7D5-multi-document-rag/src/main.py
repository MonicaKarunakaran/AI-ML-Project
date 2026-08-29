from src.config import DOCUMENTS_DIR
from src.ingestion import create_vector_index
from src.rag_pipeline import answer_question


def print_sources(sources):
    """Display the sources used to generate the answer."""

    if not sources:
        print("\nSources: No source information available.")
        return

    print("\nSources:")

    displayed = set()

    for source in sources:

        key = (
            source["source"],
            source["page"]
        )

        if key in displayed:
            continue

        displayed.add(key)

        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )


def main():
    """Run the interactive multi-document RAG application."""

    print("=" * 70)
    print("MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 70)

    print(f"\nDocuments directory:")
    print(DOCUMENTS_DIR)

    print("\nInitializing vector database...")

    create_vector_index()

    print("\nSystem ready.")
    print("Ask questions about the indexed documents.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nExiting RAG system.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            answer, sources = answer_question(question)

            print("\n" + "-" * 70)
            print("ANSWER")
            print("-" * 70)

            print(answer)

            print_sources(sources)

        except Exception as error:

            print("\nError while processing the question:")
            print(error)


if __name__ == "__main__":
    main()