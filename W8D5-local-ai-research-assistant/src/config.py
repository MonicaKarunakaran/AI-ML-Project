"""
Application configuration for the Local AI Research Assistant.
"""

from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data and output directories
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
OUTPUTS_DIR = BASE_DIR / "outputs"
VECTOR_DB_DIR = OUTPUTS_DIR / "chroma_db"

# Local Ollama configuration
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# MLflow configuration
MLFLOW_EXPERIMENT = "Local AI Research Assistant"

# Create required directories automatically
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)