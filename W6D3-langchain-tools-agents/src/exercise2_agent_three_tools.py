from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.tools.calculator_tool import calculator_tool
from src.tools.date_lookup_tool import date_lookup_tool
from src.tools.web_search_stub import web_search


def create_three_tool_agent():
    """Create an agent with three custom tools."""

    llm = ChatOllama(model="llama3.2:3b")

    tools = [
        web_search,
        calculator_tool,
        date_lookup_tool,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI assistant with access to three tools. "
            "Use the calculator tool for mathematical calculations. "
            "Use the date lookup tool for holiday/date questions. "
            "Use the web search tool for general information searches. "
            "Use previous conversation messages when answering follow-up questions. "
            "Give concise and accurate answers."
        ),
    )


def run_conversation():
    """Run three scenarios while maintaining conversation history."""

    agent = create_three_tool_agent()

    conversation_history = []

    scenarios = [
        "What is 125 * 8?",
        "What holiday is on 2026-08-15?",
        "Search for information about LangChain agents.",
    ]

    print("=" * 60)
    print("W6D3 - EXERCISE 2: MEMORY-AWARE 3-TOOL AGENT")
    print("=" * 60)

    for index, user_input in enumerate(scenarios, start=1):
        print(f"\nScenario {index}")
        print(f"User: {user_input}")

        conversation_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = agent.invoke(
                {
                    "messages": conversation_history
                }
            )

            messages = result.get("messages", [])

            # Keep the complete agent conversation for the next turn.
            conversation_history = messages

            # Display tool usage.
            for message in messages:
                if message.type == "tool":
                    print(f"Tool used: {message.name}")
                    print(f"Tool result: {message.content}")

            if messages:
                print(f"AI: {messages[-1].content}")

        except Exception as exc:
            print(f"Error: {exc}")

    print("\n" + "=" * 60)
    print("FULL CONVERSATION HISTORY")
    print("=" * 60)

    for message in conversation_history:
        if message.type in ("human", "ai"):
            role = "User" if message.type == "human" else "AI"
            print(f"{role}: {message.content}")


if __name__ == "__main__":
    run_conversation()