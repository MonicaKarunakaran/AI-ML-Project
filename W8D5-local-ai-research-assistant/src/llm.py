"""
Local LLM configuration.

This module provides the Ollama-based local language model
used by the research assistant.
"""

from langchain_ollama import ChatOllama

from src.config import LLM_MODEL, OLLAMA_BASE_URL


def get_llm() -> ChatOllama:
    """
    Create and return the local Ollama chat model.

    Returns:
        ChatOllama: Configured local Ollama language model.
    """
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )