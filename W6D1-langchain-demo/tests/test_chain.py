import pytest

from src.chain import (
    clear_memory,
    get_conversation_history,
    run_chain,
)


@pytest.fixture(autouse=True)
def reset_memory():
    clear_memory()
    yield
    clear_memory()


def test_chain_returns_dict():
    result = run_chain("What is Python?")

    assert isinstance(result, dict)


def test_chain_contains_question():
    question = "What is machine learning?"

    result = run_chain(question)

    assert "question" in result
    assert result["question"] == question


def test_chain_contains_answer():
    result = run_chain("What is RAG?")

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_chain_five_inputs():
    questions = [
        "What is AI?",
        "What is ML?",
        "What is deep learning?",
        "What is RAG?",
        "What is LangChain?",
    ]

    for question in questions:

        result = run_chain(question)

        assert isinstance(result, dict)
        assert "answer" in result


def test_memory_maintains_history():
    run_chain("My name is Monica.")

    run_chain("I am learning AI and ML.")

    history = get_conversation_history()

    assert "My name is Monica." in history
    assert "I am learning AI and ML." in history