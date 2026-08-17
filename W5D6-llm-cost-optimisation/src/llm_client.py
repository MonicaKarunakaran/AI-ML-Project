from typing import Tuple

import requests

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

from src.token_utils import (
    count_tokens,
)


class OllamaClient:
    """
    Client for local Ollama inference.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = (
            base_url.rstrip("/")
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Tuple[str, int, int]:
        """
        Generate a response using Ollama.

        Returns:

        response,
        input_tokens,
        output_tokens
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt:
            payload["system"] = (
                system_prompt
            )

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            "",
        )

        input_tokens = data.get(
            "prompt_eval_count"
        )

        output_tokens = data.get(
            "eval_count"
        )

        if input_tokens is None:
            input_tokens = count_tokens(
                prompt
            )

        if output_tokens is None:
            output_tokens = count_tokens(
                answer
            )

        return (
            answer,
            int(input_tokens),
            int(output_tokens),
        )


class MockLLMClient:
    """
    Mock client used for testing.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
    ):
        response = (
            "This is a mock response used "
            "for testing the LLM cost "
            "optimisation pipeline."
        )

        return (
            response,
            count_tokens(prompt),
            count_tokens(response),
        )