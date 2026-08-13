from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chromadb_data"

# PDF
PDF_PATH = DATA_DIR / "sample.pdf"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama model
OLLAMA_MODEL = "llama3.2:3b"

# ChromaDB collection names
DEMO_COLLECTION = "semantic_search_demo"
PDF_COLLECTION = "pdf_documents"

# Retrieval settings
TOP_K = 3