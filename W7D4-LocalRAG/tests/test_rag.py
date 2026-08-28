from pathlib import Path

from src.rag import SYSTEM_PROMPT
from src.ingest import load_documents


def test_system_prompt_exists():
    """Check that a custom system prompt is configured."""

    assert SYSTEM_PROMPT
    assert "AI/ML" in SYSTEM_PROMPT
    assert "RAG" in SYSTEM_PROMPT


def test_documents_can_be_loaded():
    """Check that the local documents are loaded successfully."""

    documents = load_documents()

    assert len(documents) >= 1


def test_sample_data_exists():
    """Check that the sample data directory contains files."""

    data_dir = Path("data/docs")

    assert data_dir.exists()
    assert any(data_dir.iterdir())
