"""
Exercise 2:
Perform cosine similarity search and metadata filtering.
"""

from pathlib import Path
import chromadb
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "w6d4_documents"
EMBEDDING_MODEL = "nomic-embed-text"


def print_results(results):
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for i, (doc_id, document, distance, metadata) in enumerate(
        zip(ids, documents, distances, metadatas),
        start=1,
    ):
        similarity = 1 - distance

        print(f"\nResult {i}")
        print(f"ID: {doc_id}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Approx. similarity: {similarity:.4f}")
        print(f"Category: {metadata['category']}")
        print(f"Document: {document}")


def main():
    print("=" * 60)
    print("W6D4 - Similarity Search + Metadata Filtering")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_collection(name=COLLECTION_NAME)

    query = "How do computers learn from data?"

    print("\n" + "-" * 60)
    print("COSINE SIMILARITY SEARCH")
    print("-" * 60)
    print(f"Query: {query}")

    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    print_results(results)

    print("\n" + "-" * 60)
    print("METADATA FILTERING")
    print("-" * 60)

    filter_query = "What technologies are used for retrieval augmented generation?"

    print(f"Query: {filter_query}")
    print("Filter: category = rag")

    query_embedding = embeddings.embed_query(filter_query)

    filtered_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"category": "rag"},
    )

    print_results(filtered_results)

    print("\n" + "=" * 60)
    print("MANUAL VERIFICATION")
    print("=" * 60)

    print(
        """
The returned documents should be semantically related to the query.

For the first query, documents related to machine learning,
AI, or learning from data should appear near the top.

For the metadata-filtered query, every returned document
should have category = rag.
"""
    )

    print("Search demonstration completed successfully.")


if __name__ == "__main__":
    main()