from src.tools.calculator import calculator
from src.tools.web_search import web_search


def test_calculator():
    result = calculator("25 * 4")
    assert result == "100"


def test_web_search_stub():
    result = web_search("LangChain")
    assert "LangChain" in result


def test_agent_creation(agent):
    assert agent is not None