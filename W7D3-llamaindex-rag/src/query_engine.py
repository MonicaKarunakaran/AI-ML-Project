import time
from pathlib import Path

from src.indexing import build_index
from src.config import OUTPUT_DIR


QUERIES = [
    "What is machine learning?",
    "What is supervised learning?",
    "What are common NLP tasks?",
    "What is NLP?",
    "What does RAG stand for?",
    "How does a RAG pipeline work?",
    "What is the purpose of vector embeddings in RAG?",
    "What is MLOps?",
    "What is MLflow used for?",
    "What is Docker used for in MLOps?",
]


def run_queries(index):
    """Run all test queries against the query engine."""

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    results = []

    for number, question in enumerate(QUERIES, start=1):

        start_time = time.perf_counter()

        response = query_engine.query(question)

        latency = (time.perf_counter() - start_time) * 1000

        answer = str(response)

        print("\n" + "=" * 70)
        print(f"Query {number}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Latency: {latency:.2f} ms")

        results.append(
            {
                "number": number,
                "question": question,
                "answer": answer,
                "latency_ms": latency,
            }
        )

    return results


def save_results(results, filename="llamaindex_results.txt"):
    """Save query results to an output file."""

    output_file = Path(OUTPUT_DIR) / filename

    with open(output_file, "w", encoding="utf-8") as file:

        for result in results:

            file.write("=" * 70 + "\n")
            file.write(f"Query {result['number']}\n")
            file.write(f"Question: {result['question']}\n")
            file.write(f"Answer: {result['answer']}\n")
            file.write(
                f"Latency: {result['latency_ms']:.2f} ms\n"
            )
            file.write("\n")

    print(f"\nResults saved to: {output_file}")


def main():
    index = build_index()

    results = run_queries(index)

    save_results(results)


if __name__ == "__main__":
    main()