from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OLLAMA_BASE_URL = "http://localhost:11434"

# Change these if your locally available Ollama models have different names.
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_COLLECTION = "w7d3_llamaindex_documents"

TOP_K = 3

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)