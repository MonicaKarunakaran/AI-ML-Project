from src.agent import create_langchain_agent, create_tools
from src.tools.calculator import calculator
from src.tools.web_search_stub import web_search


def test_tools_are_available():
    tools = create_tools()

    assert len(tools) == 2
    assert calculator in tools
    assert web_search in tools


def test_calculator_tool():
    result = calculator.invoke("25 * 8")

    assert result == "200"


def test_web_search_stub():
    result = web_search.invoke("What is RAG?")

    assert "Web search stub result" in result


def test_agent_creation():
    agent = create_langchain_agent()

    assert agent is not None