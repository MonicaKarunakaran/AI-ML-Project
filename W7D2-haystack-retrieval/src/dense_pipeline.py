from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_dense_pipeline(documents: list[Document]):
    """Create a dense retrieval pipeline using sentence embeddings."""

    document_store = InMemoryDocumentStore()

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME
    )

    document_embedder.warm_up()

    embedded_documents = document_embedder.run(documents)["documents"]

    document_store.write_documents(embedded_documents)

    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL_NAME
    )

    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        text_embedder
    )

    pipeline.add_component(
        "retriever",
        retriever
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    return pipeline


def retrieve_dense(pipeline: Pipeline, query: str):
    """Retrieve documents using dense semantic retrieval."""

    result = pipeline.run(
        {
            "text_embedder": {
                "text": query
            }
        }
    )

    return result["retriever"]["documents"]


if __name__ == "__main__":
    from src.document_loader import load_documents

    documents = load_documents()

    pipeline = create_dense_pipeline(documents)

    query = "What is the main topic of the documents?"

    results = retrieve_dense(pipeline, query)

    print("\nDENSE RETRIEVAL RESULTS")
    print("=" * 60)

    for i, document in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source: {document.meta.get('source')}")
        print(document.content[:500])