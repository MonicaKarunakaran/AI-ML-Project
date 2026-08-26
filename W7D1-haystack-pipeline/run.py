from src.config import PDF_DIR
from src.document_loader import load_documents
from src.pipeline_bm25 import run_bm25
from src.pipeline_dense import (
    prepare_embeddings,
    run_dense
)


def main():

    print("=" * 60)
    print("HAYSTACK RETRIEVAL DEMO")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    )

    # BM25

    print("\n--- BM25 RETRIEVAL ---")

    bm25_store = load_documents(
        PDF_DIR
    )

    bm25_documents = run_bm25(
        question,
        bm25_store
    )

    for index, document in enumerate(
        bm25_documents,
        start=1
    ):

        print(
            f"\nBM25 Result {index}"
        )

        print(
            document.content[:400]
        )

    # Dense

    print("\n--- DENSE RETRIEVAL ---")

    dense_store = load_documents(
        PDF_DIR
    )

    dense_store = prepare_embeddings(
        dense_store
    )

    dense_documents = run_dense(
        question,
        dense_store
    )

    for index, document in enumerate(
        dense_documents,
        start=1
    ):

        print(
            f"\nDense Result {index}"
        )

        print(
            document.content[:400]
        )


if __name__ == "__main__":
    main()