"""
Tests for the LangGraph research workflow.
"""

from langchain_core.documents import Document

from src.graph import run_research


class FakeRetriever:
    """
    Simple fake retriever used for unit testing.
    """

    def invoke(self, question: str):
        return [
            Document(
                page_content=(
                    "Artificial intelligence enables machines "
                    "to perform tasks that normally require human intelligence."
                ),
                metadata={
                    "source": "test.pdf",
                    "page": 1,
                },
            )
        ]


class FakeResponse:
    """
    Simple fake LLM response.
    """

    content = (
        "Artificial intelligence enables machines "
        "to perform tasks that normally require human intelligence."
    )


class FakeLLM:
    """
    Simple fake LLM used for unit testing.
    """

    def invoke(self, prompt):
        return FakeResponse()


def test_graph_returns_answer():
    """
    Verify that the graph produces an answer.
    """

    result = run_research(
        question="What is artificial intelligence?",
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )

    assert "answer" in result
    assert result["answer"] != ""


def test_graph_retrieves_documents():
    """
    Verify that retrieved documents are stored in graph state.
    """

    result = run_research(
        question="What is artificial intelligence?",
        retriever=FakeRetriever(),
        llm=FakeLLM(),
    )

    assert "documents" in result
    assert len(result["documents"]) == 1