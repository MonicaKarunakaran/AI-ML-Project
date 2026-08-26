import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import add_turn, create_memory, get_history


TURNS = [
    (
        "My name is Monica.",
        "Nice to meet you, Monica.",
    ),
    (
        "I am learning AI and ML.",
        "That's great. AI and ML are valuable skills.",
    ),
    (
        "What is LangChain?",
        "LangChain is a framework for building applications powered by language models.",
    ),
    (
        "What is memory used for?",
        "Memory helps maintain conversation context across multiple turns.",
    ),
    (
        "What am I learning?",
        "You are learning AI, ML, and LangChain.",
    ),
]


def main():
    memory = create_memory()

    print("=" * 60)
    print("W6D2 - CONVERSATION MEMORY DEMO")
    print("=" * 60)

    output_lines = []

    for index, (user_input, ai_response) in enumerate(TURNS, start=1):
        add_turn(
            memory,
            user_input,
            ai_response,
        )

        print(f"\n--- Turn {index} ---")
        print(f"User: {user_input}")
        print(f"AI: {ai_response}")

        output_lines.append(
            f"--- Turn {index} ---\n"
            f"User: {user_input}\n"
            f"AI: {ai_response}\n"
        )

    history = get_history(memory)

    print("\n" + "=" * 60)
    print("FINAL CONVERSATION HISTORY")
    print("=" * 60)

    output_lines.append("FINAL CONVERSATION HISTORY\n")

    for message in history:
        line = f"{message.type}: {message.content}"

        print(line)
        output_lines.append(line)

    output_file = PROJECT_ROOT / "outputs" / "memory_output.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))

    print("\n" + "=" * 60)
    print("Memory verification completed.")
    print(f"Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()