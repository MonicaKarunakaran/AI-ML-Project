from src.exercise2_agent_three_tools import create_three_tool_agent
from src.tools.calculator_tool import calculator_tool
from src.tools.date_lookup_tool import date_lookup_tool
from src.tools.web_search_stub import web_search


def test_three_tools_are_available():
    from src.exercise2_agent_three_tools import (
        calculator_tool,
        date_lookup_tool,
        web_search,
    )

    assert calculator_tool.name == "calculator_tool"
    assert date_lookup_tool.name == "date_lookup_tool"
    assert web_search.name == "web_search"


def test_calculator_tool():
    result = calculator_tool.invoke("25 * 4")

    assert result == "100"


def test_date_lookup_tool():
    result = date_lookup_tool.invoke("2026-08-15")

    assert "Independence Day" in result


def test_web_search_tool():
    result = web_search.invoke("LangChain")

    assert "LangChain" in result


def test_agent_creation():
    agent = create_three_tool_agent()

    assert agent is not None