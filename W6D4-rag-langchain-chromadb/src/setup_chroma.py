"""
Exercise 1:
Create a ChromaDB collection and add 20 documents with embeddings.
"""

from pathlib import Path
import chromadb
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "w6d4_documents"
EMBEDDING_MODEL = "nomic-embed-text"


DOCUMENTS = [
    "Machine learning allows computers to learn patterns from data.",
    "Artificial intelligence enables machines to perform tasks that normally require human intelligence.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Natural language processing helps computers understand and generate human language.",
    "Computer vision enables machines to interpret images and videos.",
    "Python is widely used for machine learning and data science.",
    "Data preprocessing includes cleaning, transforming, and preparing data for models.",
    "Feature engineering creates useful input variables from raw data.",
    "Supervised learning uses labeled data to train predictive models.",
    "Unsupervised learning finds patterns in data without labeled examples.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Vector embeddings represent text as numerical vectors.",
    "Semantic search retrieves information based on meaning rather than exact keywords.",
    "Retrieval augmented generation combines document retrieval with language generation.",
    "ChromaDB is a vector database commonly used for AI and RAG applications.",
    "Large language models can generate natural language responses from prompts.",
    "Cosine similarity measures the similarity between vector directions.",
    "Metadata can be used to filter and organize documents in a vector database.",
    "RAG systems retrieve relevant context before asking an LLM to generate an answer.",
    "Ollama allows large language models to run locally on a computer.",
]


def main():
    print("=" * 60)
    print("W6D4 - ChromaDB Vector Store Setup")
    print("=" * 60)

    print("\nLoading embedding model...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Connecting to ChromaDB: {CHROMA_DIR}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection removed.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("\nGenerating embeddings for 20 documents...")

    vectors = embeddings.embed_documents(DOCUMENTS)

    ids = [f"doc_{i + 1}" for i in range(len(DOCUMENTS))]

    metadatas = [
        {
            "category": (
                "machine_learning"
                if i < 5
                else "data_science"
                if i < 10
                else "rag"
                if i < 15
                else "ai"
            ),
            "source": "w6d4_demo",
        }
        for i in range(len(DOCUMENTS))
    ]

    collection.add(
        ids=ids,
        documents=DOCUMENTS,
        embeddings=vectors,
        metadatas=metadatas,
    )

    print("\nSuccessfully added 20 documents.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Document count: {collection.count()}")
    print("\nSample stored document:")
    print(collection.get(ids=["doc_1"])["documents"][0])
    print("\nExercise 1 completed successfully.")


if __name__ == "__main__":
    main()