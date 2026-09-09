from src.agents import (
    create_researcher,
    create_writer,
    create_reviewer,
)

from src.tasks import (
    create_research_task,
    create_writing_task,
    create_review_task,
)


def test_researcher_agent():
    agent = create_researcher()

    assert agent.role == "Researcher"
    assert agent.goal
    assert agent.backstory


def test_writer_agent():
    agent = create_writer()

    assert agent.role == "Technical Writer"
    assert agent.goal
    assert agent.backstory


def test_reviewer_agent():
    agent = create_reviewer()

    assert agent.role == "Quality Reviewer"
    assert agent.goal
    assert agent.backstory


def test_research_task():
    researcher = create_researcher()

    task = create_research_task(researcher)

    assert task.agent == researcher
    assert task.description
    assert task.expected_output


def test_writing_task():
    writer = create_writer()

    task = create_writing_task(writer)

    assert task.agent == writer
    assert task.description
    assert task.expected_output


def test_review_task():
    reviewer = create_reviewer()

    task = create_review_task(reviewer)

    assert task.agent == reviewer
    assert task.description
    assert task.expected_output