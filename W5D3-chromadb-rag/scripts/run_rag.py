import sys
from src.rag_pipeline import run_rag

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/run_rag.py "
            "<pdf_path> <question>"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = " ".join(sys.argv[2:])

    result = run_rag(
        pdf_path=pdf_path,
        question=question
    )

    print("\nQuestion")
    print("=" * 60)
    print(result["question"])

    print("\nRetrieved Top-3 Chunks")
    print("=" * 60)

    for i, chunk in enumerate(
        result["retrieved_chunks"]
    ):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk[:1000])

    print("\nGenerated Answer")
    print("=" * 60)
    print(result["answer"])


if __name__ == "__main__":
    main()