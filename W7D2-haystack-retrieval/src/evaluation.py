import csv

from src.document_loader import load_documents
from src.bm25_pipeline import create_bm25_pipeline, retrieve_bm25
from src.dense_pipeline import create_dense_pipeline, retrieve_dense


QUESTIONS = [
    "What is the main topic discussed in the documents?",
    "What are the key concepts explained?",
    "What are the important definitions?",
    "What methods or approaches are described?",
    "What are the main benefits discussed?",
    "What are the major challenges mentioned?",
    "What examples are provided?",
    "What are the important steps explained?",
    "What conclusions are presented?",
    "What are the main recommendations?"
]


def evaluate_retrieval():
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("\nCreating BM25 pipeline...")
    bm25_pipeline = create_bm25_pipeline(documents)

    print("Creating Dense pipeline...")
    dense_pipeline = create_dense_pipeline(documents)

    results = []

    for question in QUESTIONS:

        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        bm25_results = retrieve_bm25(
            bm25_pipeline,
            question
        )

        dense_results = retrieve_dense(
            dense_pipeline,
            question
        )

        bm25_sources = [
            doc.meta.get("source", "unknown")
            for doc in bm25_results
        ]

        dense_sources = [
            doc.meta.get("source", "unknown")
            for doc in dense_results
        ]

        print("\nBM25:")
        print(bm25_sources)

        print("\nDense:")
        print(dense_sources)

        bm25_relevant = input(
            "Is BM25 retrieval relevant? (y/n): "
        ).strip().lower()

        dense_relevant = input(
            "Is Dense retrieval relevant? (y/n): "
        ).strip().lower()

        results.append(
            {
                "question": question,
                "bm25_sources": ", ".join(bm25_sources),
                "dense_sources": ", ".join(dense_sources),
                "bm25_relevant": bm25_relevant,
                "dense_relevant": dense_relevant,
            }
        )

    output_file = "outputs/retrieval_comparison.csv"

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation saved to: {output_file}")


if __name__ == "__main__":
    evaluate_retrieval()