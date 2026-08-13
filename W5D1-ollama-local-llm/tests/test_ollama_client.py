from src.ollama_client import chat_with_ollama


def test_ollama_returns_string():
    response = chat_with_ollama(
        prompt="What is Python?",
        model="llama3.2:3b",
    )

    assert isinstance(response, str)
    assert len(response.strip()) > 0