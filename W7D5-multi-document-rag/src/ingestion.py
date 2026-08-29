import chromadb

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.loader import load_documents


def configure_embedding_model():
    """Configure the Ollama embedding model."""

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL
    )


def configure_chunking():
    """Configure document chunking."""

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def get_chroma_collection(reset=False):
    """
    Create or retrieve the persistent ChromaDB collection.

    Args:
        reset: If True, delete the existing collection first.

    Returns:
        Chroma collection.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing ChromaDB collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def create_vector_index(reset=False):
    """
    Create a vector index from all PDF documents.

    The documents are:
        1. Loaded
        2. Split into chunks
        3. Embedded using Ollama
        4. Stored in ChromaDB

    Args:
        reset: Rebuild the vector database if True.

    Returns:
        VectorStoreIndex
    """

    configure_embedding_model()
    configure_chunking()

    collection = get_chroma_collection(reset=reset)

    # If an existing collection already contains vectors,
    # reuse it instead of embedding everything again.
    if collection.count() > 0 and not reset:
        print(
            f"Using existing ChromaDB collection "
            f"with {collection.count()} vectors."
        )

        vector_store = ChromaVectorStore(
            chroma_collection=collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

    # Load all PDFs
    documents = load_documents()

    # Create vector store
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("\nCreating embeddings and storing vectors...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(
        f"\nChromaDB now contains "
        f"{collection.count()} vectors."
    )

    return index