from src.chains.doc_qa_chain import get_chain, ask_question
from src.chains.agent import create_chatbot_agent, run_agent_task


def run_chain_demo():
    print("\n" + "=" * 60)
    print("CHAIN DEMO - 5 INPUTS")
    print("=" * 60)

    chain, memory = get_chain()

    questions = [
        "What is machine learning?",
        "What is RAG in AI?",
        "What are embeddings?",
        "What is LangChain?",
        "What is an AI agent?",
    ]

    for index, question in enumerate(questions, start=1):
        print(f"\nInput {index}: {question}")

        response = ask_question(
            chain,
            memory,
            question,
        )

        print("Answer:", response["answer"])
        print("Sources:", response["sources"])


def run_memory_demo():
    print("\n" + "=" * 60)
    print("MEMORY DEMO - 5 TURNS")
    print("=" * 60)

    chain, memory = get_chain()

    turns = [
        "My name is Monica.",
        "I am learning Python.",
        "I am learning LangChain.",
        "I am interested in AI and machine learning.",
        "What have I told you about myself?",
    ]

    for index, question in enumerate(turns, start=1):
        print(f"\nTurn {index}: {question}")

        response = ask_question(
            chain,
            memory,
            question,
        )

        print("Response:", response["answer"])

    history = memory.load_memory_variables({}).get(
        "history",
        "",
    )

    print("\nStored conversation history:")
    print(history)


def run_agent_demo():
    print("\n" + "=" * 60)
    print("AGENT DEMO - 3 TASKS")
    print("=" * 60)

    agent = create_chatbot_agent()

    tasks = [
        "Calculate 25 * 4.",
        "Calculate 144 / 12.",
        "Search for information about LangChain.",
    ]

    for index, task in enumerate(tasks, start=1):
        print(f"\nTask {index}: {task}")

        result = run_agent_task(
            agent,
            task,
        )

        messages = result.get("messages", [])

        if messages:
            final_message = messages[-1]
            print("Response:", final_message.content)
        else:
            print("Result:", result)


def main():
    print("=" * 60)
    print("W6D5 - DOCUMENT CHATBOT WITH LANGCHAIN")
    print("=" * 60)

    run_chain_demo()
    run_memory_demo()
    run_agent_demo()

    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()