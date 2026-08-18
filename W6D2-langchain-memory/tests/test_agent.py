from src.tools import TOOLS


def test_agent_has_two_tools():
    assert len(TOOLS) == 2


def test_calculator_tool():
    calculator = next(
        tool for tool in TOOLS
        if tool.name == "calculator"
    )

    result = calculator.invoke("25 * 4")

    assert result == "100"


def test_web_search_tool():
    search = next(
        tool for tool in TOOLS
        if tool.name == "web_search"
    )

    result = search.invoke("LangChain")

    assert "Search result for" in result
    assert "LangChain" in result