from src.embedding import get_embedding

def test_embedding_returns_vector():
    embedding = get_embedding(
        "Machine learning is useful."
    )

    assert isinstance(embedding, list)
    assert len(embedding) > 0