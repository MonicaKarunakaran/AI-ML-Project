import requests


OLLAMA_API_URL = "http://localhost:11434/api/chat"


def chat_with_ollama(
    prompt: str,
    model: str = "llama3.2:3b",
    system_prompt: str = "",
    temperature: float = 0.7,
) -> str:
    """Send a chat request to the local Ollama API."""

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        return data["message"]["content"]

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Ollama: {exc}"
        ) from exc