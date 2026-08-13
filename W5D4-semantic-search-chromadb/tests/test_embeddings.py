from src.embeddings import EmbeddingModel


def test_embedding_dimension():

    model = EmbeddingModel()

    vector = model.embed(
        "This is a test sentence."
    )

    assert len(vector) == 384


def test_embedding_returns_numbers():

    model = EmbeddingModel()

    vector = model.embed(
        "Semantic search."
    )

    assert isinstance(vector, list)
    assert all(
        isinstance(value, float)
        for value in vector
    )