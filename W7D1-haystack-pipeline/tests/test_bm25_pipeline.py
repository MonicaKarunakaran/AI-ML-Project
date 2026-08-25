from src.config import PDF_DIR
from src.document_loader import load_documents
from src.pipeline_bm25 import build_bm25_pipeline


def test_document_store_contains_chunks():

    store = load_documents(PDF_DIR)

    assert store.count_documents() > 0


def test_bm25_pipeline_builds():

    store = load_documents(PDF_DIR)

    pipeline = build_bm25_pipeline(store)

    assert pipeline is not None