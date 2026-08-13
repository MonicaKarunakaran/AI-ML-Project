from typing import List

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Wrapper around the SentenceTransformer embedding model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """Generate an embedding for one text."""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts).tolist()


if __name__ == "__main__":
    embedder = EmbeddingModel()

    text = "Semantic search finds information based on meaning."

    vector = embedder.embed(text)

    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding dimension:", len(vector))
    print("First 5 values:", vector[:5])