from src.chromadb_client import (
    ChromaDBManager,
    create_demo_documents,
)


def test_demo_document_count(tmp_path):

    manager = ChromaDBManager(
        persist_directory=tmp_path,
        collection_name="test_collection",
    )

    documents, ids, metadatas = (
        create_demo_documents()
    )

    manager.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    assert manager.count() == 20


def test_similarity_search(tmp_path):

    manager = ChromaDBManager(
        persist_directory=tmp_path,
        collection_name="search_collection",
    )

    documents, ids, metadatas = (
        create_demo_documents()
    )

    manager.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    results = manager.similarity_search(
        "machine learning models",
        n_results=3,
    )

    assert len(results["documents"][0]) == 3


def test_metadata_filter(tmp_path):

    manager = ChromaDBManager(
        persist_directory=tmp_path,
        collection_name="filter_collection",
    )

    documents, ids, metadatas = (
        create_demo_documents()
    )

    manager.add_documents(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    results = manager.metadata_filter(
        {"topic": "Machine Learning"}
    )

    assert len(results["documents"]) > 0