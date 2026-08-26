from src.agent import create_langchain_agent


def main():
    tasks = [
        "Calculate 125 * 24.",
        "Calculate (450 / 9) + 25.",
        "Search for information about Retrieval-Augmented Generation.",
    ]

    agent = create_langchain_agent()

    print("=" * 60)
    print("W6D3 - LANGCHAIN TOOLS & AGENT")
    print("=" * 60)

    for i, task in enumerate(tasks, start=1):
        print(f"\nTask {i}: {task}")

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": task,
                        }
                    ]
                }
            )

            messages = result.get("messages", [])

            print("\nAgent execution:")

            for message in messages:
                if message.type == "tool":
                    print(f"Tool used: {message.name}")
                    print(f"Tool result: {message.content}")

            if messages:
                print(f"\nFinal answer: {messages[-1].content}")

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()