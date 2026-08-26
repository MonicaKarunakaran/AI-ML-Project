import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import create_agent


TASKS = [
    "Use the calculator tool to calculate 25 * 4.",
    "Use the web_search tool to search for information about LangChain.",
    "Use the calculator tool to calculate 100 divided by 4.",
]

def main():
    agent = create_agent()

    print("=" * 60)
    print("W6D2 - LANGCHAIN AGENT DEMO")
    print("=" * 60)

    output_lines = []

    for index, task in enumerate(TASKS, start=1):
        print("\n" + "=" * 60)
        print(f"TASK {index}")
        print("=" * 60)
        print(f"Input: {task}")

        output_lines.append("=" * 60)
        output_lines.append(f"TASK {index}")
        output_lines.append("=" * 60)
        output_lines.append(f"Input: {task}")

        try:
            result = agent.invoke({"input": task})

            if isinstance(result, dict):
                answer = result.get("output", str(result))
            else:
                answer = str(result)

            print(f"\nFinal Answer: {answer}")
            output_lines.append(f"Final Answer: {answer}")

        except Exception as exc:
            print(f"\nAgent error: {exc}")
            output_lines.append(f"Agent error: {exc}")

    output_file = PROJECT_ROOT / "outputs" / "agent_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))

    print("\n" + "=" * 60)
    print("Agent execution completed.")
    print(f"Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()