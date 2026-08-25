from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from src.config import TOP_K
from src.document_loader import load_documents
from src.config import PDF_DIR


def build_bm25_pipeline(document_store):

    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=TOP_K
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "retriever",
        retriever
    )

    return pipeline


def run_bm25(question, document_store):

    pipeline = build_bm25_pipeline(document_store)

    result = pipeline.run(
        {
            "retriever": {
                "query": question
            }
        }
    )

    return result["retriever"]["documents"]


if __name__ == "__main__":

    document_store = load_documents(PDF_DIR)

    question = "What is the main topic discussed in the documents?"

    documents = run_bm25(
        question,
        document_store
    )

    print("\nBM25 RESULTS")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\nResult {i}")
        print("-" * 40)
        print(document.content[:500])