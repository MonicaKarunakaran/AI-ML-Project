from src.agent import run_agent


def test_agent_calculator():

    result = run_agent(
        "Calculate 25 * 16."
    )

    assert isinstance(result, dict)
    assert "input" in result
    assert "output" in result
    assert len(result["output"]) > 0


def test_agent_search():

    result = run_agent(
        "Search for the placement statistics of Data Science."
    )

    assert isinstance(result, dict)
    assert "output" in result
    assert len(result["output"]) > 0


def test_agent_combined():

    result = run_agent(
        "Calculate (100 + 50) / 5."
    )

    assert isinstance(result, dict)
    assert "input" in result
    assert "output" in result