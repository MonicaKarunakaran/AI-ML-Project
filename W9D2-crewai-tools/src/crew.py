from crewai import Crew, Process

from src.agents import researcher, writer, reviewer
from src.tasks import research_task, writing_task, review_task


research_crew = Crew(
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


def run_research(topic: str):
    """Run the multi-agent research crew."""

    result = research_crew.kickoff(
        inputs={
            "topic": topic
        }
    )

    return result


if __name__ == "__main__":
    topic = "Applications of Generative AI in Healthcare"

    result = run_research(topic)

    print("\n" + "=" * 80)
    print("FINAL RESEARCH REPORT")
    print("=" * 80)
    print(result)

    with open(
        "outputs/research_report.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(result))

    print("\nReport saved to outputs/research_report.md")