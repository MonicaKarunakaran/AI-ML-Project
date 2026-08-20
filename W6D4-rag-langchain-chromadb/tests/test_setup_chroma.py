from src.setup_chroma import DOCUMENTS


def test_document_count():
    assert len(DOCUMENTS) == 20


def test_documents_are_not_empty():
    assert all(document.strip() for document in DOCUMENTS)