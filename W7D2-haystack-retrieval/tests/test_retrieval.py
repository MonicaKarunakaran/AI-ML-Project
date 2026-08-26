from src.document_loader import load_documents
from src.bm25_pipeline import (
    create_bm25_pipeline,
    retrieve_bm25,
)


def test_documents_are_loaded():
    documents = load_documents("data")

    assert len(documents) == 5


def test_bm25_retrieval():
    documents = load_documents("data")

    pipeline = create_bm25_pipeline(documents)

    results = retrieve_bm25(
        pipeline,
        "What is the main topic?"
    )

    assert len(results) > 0


def test_retrieved_documents_have_content():
    documents = load_documents("data")

    pipeline = create_bm25_pipeline(documents)

    results = retrieve_bm25(
        pipeline,
        "What is discussed?"
    )

    assert results[0].content