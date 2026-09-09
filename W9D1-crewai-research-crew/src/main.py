import argparse
from pathlib import Path

from src.crew import create_crew


OUTPUT_DIR = Path("outputs")


def run_research(topic, use_web_search=False):
    """
    Run the CrewAI research crew.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("CREWAI MULTI-AGENT RESEARCH CREW")
    print("=" * 70)

    print(f"\nTopic: {topic}")

    if use_web_search:
        print("Mode: Web Search Enabled")
    else:
        print("Mode: Basic Research - No Web Search")

    print("\nAgents:")
    print("1. Researcher")
    print("2. Writer")
    print("3. Reviewer")

    print("\nWorkflow:")
    print("Researcher -> Writer -> Reviewer")

    print("\nStarting CrewAI...\n")

    crew = create_crew(use_web_search=use_web_search)

    result = crew.kickoff(
        inputs={
            "topic": topic
        }
    )

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(result)

    # Save output
    if use_web_search:
        output_file = OUTPUT_DIR / "research_with_web.txt"
    else:
        output_file = OUTPUT_DIR / "research_without_web.txt"

    output_file.write_text(
        str(result),
        encoding="utf-8"
    )

    print(f"\nOutput saved to: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="CrewAI Multi-Agent Research Crew"
    )

    parser.add_argument(
        "--topic",
        default="How Generative AI is changing software development",
        help="Research topic"
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web search"
    )

    args = parser.parse_args()

    run_research(
        topic=args.topic,
        use_web_search=args.web
    )


if __name__ == "__main__":
    main()