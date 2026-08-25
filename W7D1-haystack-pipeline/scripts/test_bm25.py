from src.config import PDF_DIR
from src.document_loader import load_documents
from src.pipeline_bm25 import run_bm25


def main():

    print("=" * 60)
    print("HAYSTACK BM25 RETRIEVAL TEST")
    print("=" * 60)

    document_store = load_documents(
        PDF_DIR
    )

    question = input(
        "\nEnter your question: "
    )

    documents = run_bm25(
        question,
        document_store
    )

    print("\nTOP RESULTS")
    print("=" * 60)

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(f"\nResult {index}")
        print("-" * 40)

        print(
            document.content[:700]
        )


if __name__ == "__main__":
    main()