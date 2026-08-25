from haystack import Pipeline
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever,
)

from src.config import (
    EMBEDDING_MODEL,
    TOP_K,
)


def prepare_embeddings(document_store):
    """Generate embeddings for all documents."""

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )

    document_embedder.warm_up()

    documents = document_store.filter_documents()

    result = document_embedder.run(
        documents=documents
    )

    embedded_documents = result["documents"]

    # Write embeddings back to the store.
    document_store.delete_all_documents()
    document_store.write_documents(embedded_documents)

    return document_store


def build_dense_pipeline(document_store):
    """Build the dense retrieval pipeline."""

    text_embedder = SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    )

    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        text_embedder,
    )

    pipeline.add_component(
        "retriever",
        retriever,
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding",
    )

    return pipeline


def run_dense(question, document_store):
    """Run dense retrieval for a question."""

    pipeline = build_dense_pipeline(
        document_store
    )

    result = pipeline.run(
        {
            "text_embedder": {
                "text": question
            }
        }
    )

    return result["retriever"]["documents"]