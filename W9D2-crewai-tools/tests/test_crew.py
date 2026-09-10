from src.agents import researcher, writer, reviewer
from src.tasks import research_task, writing_task, review_task
from src.crew import research_crew


def test_agents_exist():
    assert researcher is not None
    assert writer is not None
    assert reviewer is not None


def test_tasks_exist():
    assert research_task is not None
    assert writing_task is not None
    assert review_task is not None


def test_crew_configuration():
    assert len(research_crew.agents) == 3
    assert len(research_crew.tasks) == 3


def test_agent_roles():
    assert researcher.role == "AI Researcher"
    assert writer.role == "Technical Writer"
    assert reviewer.role == "Research Reviewer"