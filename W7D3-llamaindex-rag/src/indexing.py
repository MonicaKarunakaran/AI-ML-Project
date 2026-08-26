from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.config import (
    DATA_DIR,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
)


def configure_llamaindex():
    """Configure Ollama LLM and embedding model."""

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

    documents = reader.load_data()

    print(f"Loaded {len(documents)} documents.")

    return documents


def build_index():
    """Build a LlamaIndex VectorStoreIndex."""

    configure_llamaindex()

    documents = load_documents()

    index = VectorStoreIndex.from_documents(documents)

    print("LlamaIndex VectorStoreIndex created successfully.")

    return index


if __name__ == "__main__":
    build_index()