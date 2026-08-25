from src.config import PDF_DIR
from src.document_loader import load_documents

from src.pipeline_dense import (
    prepare_embeddings,
    build_dense_pipeline,
)


def test_dense_embeddings_are_created():

    store = load_documents(PDF_DIR)

    store = prepare_embeddings(store)

    assert store.count_documents() > 0


def test_dense_pipeline_builds():

    store = load_documents(PDF_DIR)

    store = prepare_embeddings(store)

    pipeline = build_dense_pipeline(store)

    assert pipeline is not None