import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

MEMORY_K = int(os.getenv("MEMORY_K", "5"))
TOP_K = int(os.getenv("TOP_K", "5"))

RAW_DATA_DIR = BASE_DIR / "src" / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "src" / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)