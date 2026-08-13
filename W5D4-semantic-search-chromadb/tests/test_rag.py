from src.chromadb_client import (
    ChromaDBManager,
    create_demo_documents,
)

from src.rag import RAGPipeline


def test_rag_collection_retrieval(tmp_path):

    manager = ChromaDBManager(
        persist_directory=tmp_path,
        collection_name="rag_test",
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
        "What is machine learning?",
        n_results=3,
    )

    assert len(results["documents"][0]) == 3