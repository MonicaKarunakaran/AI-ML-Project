from crewai import Crew, Process

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


def create_crew(use_web_search=False):
    """
    Create the complete three-agent research crew.

    Workflow:

    Researcher -> Writer -> Reviewer
    """

    tools = []

    if use_web_search:
        from src.tools import get_web_search_tool

        search_tool = get_web_search_tool()

        if search_tool:
            tools.append(search_tool)

    # Create agents
    researcher = create_researcher(tools=tools)

    writer = create_writer()

    reviewer = create_reviewer()

    # Create tasks
    research_task = create_research_task(researcher)

    writing_task = create_writing_task(writer)

    review_task = create_review_task(reviewer)

    # Create Crew
    crew = Crew(
        agents=[
            researcher,
            writer,
            reviewer,
        ],
        tasks=[
            research_task,
            writing_task,
            review_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew