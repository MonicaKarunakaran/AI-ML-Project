from src.memory import run_conversation


def main():
    results, history = run_conversation()

    print("=" * 60)
    print("W6D3 - CONVERSATION MEMORY")
    print("=" * 60)

    for i, (user_input, response) in enumerate(results, start=1):
        print(f"\nTurn {i}")
        print(f"User: {user_input}")
        print(f"AI: {response}")

    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY")
    print("=" * 60)

    for message in history:
        role = "User" if message.type == "human" else "AI"
        print(f"{role}: {message.content}")


if __name__ == "__main__":
    main()