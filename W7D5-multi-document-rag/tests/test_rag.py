from src.rag_pipeline import answer_question


def test_rag_generates_answer():
    """Verify that the RAG pipeline generates an answer."""

    answer, sources = answer_question(
        "What is Newton's second law?"
    )

    assert answer
    assert isinstance(answer, str)

    assert sources
    assert isinstance(sources, list)


def test_rag_handles_multiple_documents():
    """Verify that the RAG pipeline can answer another domain question."""

    answer, sources = answer_question(
        "What is photosynthesis?"
    )

    assert answer
    assert isinstance(answer, str)

    assert sources