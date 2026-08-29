from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Ollama configuration
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

# Chroma collection name
COLLECTION_NAME = "multi_document_rag"