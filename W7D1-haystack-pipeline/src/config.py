from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "questions.txt"

OUTPUT_DIR = BASE_DIR / "outputs"

DOCUMENT_STORE_PATH = OUTPUT_DIR / "documents"

TOP_K = 5

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"