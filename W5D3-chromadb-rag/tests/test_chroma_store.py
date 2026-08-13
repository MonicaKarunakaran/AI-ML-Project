from src.chroma_store import ChromaStore

def test_chroma_collection(tmp_path):
    store = ChromaStore(
        persist_directory=str(tmp_path),
        collection_name="test_collection"
    )

    store.add_documents(
        documents=["Python is a programming language."],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{"category": "programming"}],
        ids=["test_1"]
    )

    assert store.count() == 1