from pathlib import Path
from src.embedding import get_embeddings
from src.chroma_store import ChromaStore

def load_documents():
    path = Path("data/sample_docs.txt")

    with open(path, "r", encoding="utf-8") as file:
        documents = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return documents

def main():
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")
    embeddings = get_embeddings(documents)
    metadatas = [
        {
            "source": "sample_docs.txt",
            "category": "machine_learning"
        }
        for _ in documents
    ]

    ids = [f"doc_{i+1}" for i in range(len(documents))]

    store = ChromaStore()
    store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Documents stored in ChromaDB: {store.count()}")
    query = "How are text embeddings used in vector databases?"
    query_embedding = get_embeddings([query])[0]
    results = store.query(
        query_embedding=query_embedding,
        n_results=3
    )

    print("\nSimilarity Search Results")
    print("=" * 60)

    for i, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        print(f"\nRank: {i + 1}")
        print(f"Document: {document}")
        print(f"Distance: {distance:.4f}")

    print("\nMetadata Filter Results")
    print("=" * 60)

    filtered_results = store.query(
        query_embedding=query_embedding,
        n_results=3,
        where={"category": "machine_learning"}
    )

    for i, document in enumerate(
        filtered_results["documents"][0]
    ):
        print(f"\n{i + 1}. {document}")


if __name__ == "__main__":
    main()