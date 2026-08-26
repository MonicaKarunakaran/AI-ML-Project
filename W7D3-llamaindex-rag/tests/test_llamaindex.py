from pathlib import Path

from src.config import DATA_DIR


def test_data_directory_exists():
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()


def test_text_documents_exist():
    documents = list(DATA_DIR.glob("*.txt"))

    assert len(documents) >= 3


def test_text_documents_are_not_empty():
    documents = list(DATA_DIR.glob("*.txt"))

    assert documents

    for document in documents:
        assert document.read_text(encoding="utf-8").strip()