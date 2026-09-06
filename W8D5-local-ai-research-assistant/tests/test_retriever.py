"""
Tests for retrieval-related helper functions.
"""

from langchain_core.documents import Document

from src.retriever import format_documents


def test_format_documents():
    """
    Verify that retrieved documents are formatted correctly.
    """

    documents = [
        Document(
            page_content="Artificial intelligence is a field of computer science.",
            metadata={
                "source": "research.pdf",
                "page": 2,
            },
        )
    ]

    result = format_documents(documents)

    assert "Artificial intelligence" in result
    assert "research.pdf" in result
    assert "page 2" in result