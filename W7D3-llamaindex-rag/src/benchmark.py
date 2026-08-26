import time
from statistics import mean

from src.query_engine import QUERIES
from src.chroma_index import build_chroma_index


def benchmark_chromadb():

    index = build_chroma_index()

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    latencies = []

    print("\n" + "=" * 70)
    print("CHROMADB QUERY BENCHMARK")
    print("=" * 70)

    for number, question in enumerate(QUERIES, start=1):

        start_time = time.perf_counter()

        response = query_engine.query(question)

        latency = (time.perf_counter() - start_time) * 1000

        latencies.append(latency)

        print(f"\nQuery {number}: {question}")
        print(f"Answer: {response}")
        print(f"Latency: {latency:.2f} ms")

    average_latency = mean(latencies)

    print("\n" + "=" * 70)
    print(f"Average ChromaDB latency: {average_latency:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_chromadb()