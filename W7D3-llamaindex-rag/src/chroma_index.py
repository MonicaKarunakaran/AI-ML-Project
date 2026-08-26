import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    DATA_DIR,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION,
)


def configure_llamaindex():
    """Configure Ollama models."""

    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_documents():
    """Load documents from the data directory."""

    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
    )

    return reader.load_data()


def build_chroma_index():
    """Build a LlamaIndex index backed by ChromaDB."""

    configure_llamaindex()

    documents = load_documents()

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    chroma_collection = chroma_client.get_or_create_collection(
        CHROMA_COLLECTION
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    print("LlamaIndex + ChromaDB index created successfully.")

    return index


if __name__ == "__main__":
    build_chroma_index()