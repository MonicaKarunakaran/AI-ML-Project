from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


def create_bm25_pipeline(documents: list[Document]):
    """Create a Haystack BM25 retrieval pipeline."""

    document_store = InMemoryDocumentStore()

    document_store.write_documents(documents)

    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "retriever",
        retriever
    )

    return pipeline


def retrieve_bm25(
    pipeline: Pipeline,
    query: str
):
    """Retrieve documents using BM25."""

    result = pipeline.run(
        {
            "retriever": {
                "query": query
            }
        }
    )

    return result["retriever"]["documents"]


if __name__ == "__main__":
    from src.document_loader import load_documents

    documents = load_documents()

    pipeline = create_bm25_pipeline(documents)

    query = "What is the main topic of the documents?"

    results = retrieve_bm25(pipeline, query)

    print("\nBM25 RESULTS")
    print("=" * 60)

    for i, document in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source: {document.meta.get('source')}")
        print(document.content[:500])