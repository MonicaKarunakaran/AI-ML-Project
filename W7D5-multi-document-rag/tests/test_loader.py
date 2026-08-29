from src.loader import get_pdf_files, load_documents


def test_pdf_files_exist():
    """Verify that multiple PDF files are available."""

    pdf_files = get_pdf_files()

    assert len(pdf_files) >= 2


def test_load_multiple_documents():
    """Verify that multiple PDF documents can be loaded."""

    documents = load_documents()

    assert documents
    assert len(documents) > 1


def test_documents_have_metadata():
    """Verify that loaded documents contain metadata."""

    documents = load_documents()

    assert documents

    for document in documents:
        assert document.metadata is not None