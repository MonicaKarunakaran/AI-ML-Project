from typing import List, Dict, Any

import chromadb

from src.config import CHROMA_DIR, DEMO_COLLECTION
from src.embeddings import EmbeddingModel


class ChromaDBManager:
    """Manage ChromaDB collections and semantic search."""

    def __init__(
        self,
        persist_directory=CHROMA_DIR,
        collection_name=DEMO_COLLECTION,
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Semantic search demonstration"
            },
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

        self.embedder = EmbeddingModel()

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add documents and generated embeddings to ChromaDB."""

        embeddings = self.embedder.embed_documents(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict[str, Any] | None = None,
    ):
        """Perform cosine similarity search."""

        query_embedding = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return results

    def metadata_filter(self, where: Dict[str, Any]):
        """Retrieve documents using metadata filtering."""

        return self.collection.get(
            where=where,
            include=[
                "documents",
                "metadatas",
            ],
        )

    def count(self) -> int:
        """Return number of documents in collection."""

        return self.collection.count()


def create_demo_documents():
    """Create 20 realistic AI/ML documents."""

    topics = [
        (
            "Machine Learning",
            "Machine learning enables computers to learn patterns from data and make predictions."
        ),
        (
            "Deep Learning",
            "Deep learning uses neural networks with multiple layers to learn complex representations."
        ),
        (
            "Natural Language Processing",
            "Natural language processing allows computers to understand and process human language."
        ),
        (
            "Computer Vision",
            "Computer vision enables machines to understand images and visual information."
        ),
        (
            "MLOps",
            "MLOps combines machine learning development with deployment, monitoring, and automation."
        ),
    ]

    documents = []
    ids = []
    metadatas = []

    for i in range(20):
        topic, text = topics[i % len(topics)]

        document = (
            f"Document {i + 1}: {text} "
            f"This document provides an introduction to {topic.lower()} "
            f"and explains how it is used in practical AI systems."
        )

        documents.append(document)
        ids.append(f"doc_{i + 1}")

        metadatas.append(
            {
                "topic": topic,
                "author": f"user_{(i % 3) + 1}",
                "document_type": "training",
            }
        )

    return documents, ids, metadatas


def main():
    print("=" * 60)
    print("W5D4 - ChromaDB Semantic Search")
    print("=" * 60)

    manager = ChromaDBManager()

    documents, ids, metadatas = create_demo_documents()

    manager.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"\nDocuments in collection: {manager.count()}")

    # ---------------------------------------------------------
    # Similarity search
    # ---------------------------------------------------------

    query = "How can computers learn patterns from information?"

    print("\n" + "-" * 60)
    print("SEMANTIC SEARCH")
    print("-" * 60)

    results = manager.similarity_search(
        query=query,
        n_results=5,
    )

    for i in range(len(results["documents"][0])):
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")
        print("Topic:", metadata["topic"])
        print("Author:", metadata["author"])
        print("Cosine distance:", round(distance, 4))
        print("Document:", document)

    # ---------------------------------------------------------
    # Metadata filtering
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("METADATA FILTERING")
    print("-" * 60)

    filtered = manager.metadata_filter(
        {"topic": "Machine Learning"}
    )

    for document, metadata in zip(
        filtered["documents"],
        filtered["metadatas"],
    ):
        print(
            f"\nTopic: {metadata['topic']}"
            f"\nAuthor: {metadata['author']}"
            f"\n{document}"
        )


if __name__ == "__main__":
    main()