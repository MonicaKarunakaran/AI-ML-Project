import ollama

def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:

    response = ollama.embeddings(
        model=model,
        prompt=text
    )

    return response["embedding"]

def get_embeddings(
    texts: list[str],
    model: str = "nomic-embed-text"
) -> list[list[float]]:

    return [get_embedding(text, model) for text in texts]