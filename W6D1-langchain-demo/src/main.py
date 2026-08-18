import json
import time

import mlflow

from .agent import run_agent
from .chain import (
    clear_memory,
    get_conversation_history,
    run_chain,
)


MLFLOW_DB = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_DB)

mlflow.set_experiment(
    "W6D1-LangChain-Fundamentals"
)

mlflow.set_experiment(
    "W6D1-LangChain-Fundamentals"
)


CHAIN_QUESTIONS = [
    "What is machine learning?",
    "What is deep learning?",
    "What is RAG?",
    "What is an LLM?",
    "What is LangChain?",
]


AGENT_TASKS = [
    "Calculate 25 * 16.",
    "Search for the placement statistics of Data Science.",
    "Calculate (100 + 50) / 5 and explain the result.",
]


def run_chain_demo():

    print("\n")
    print("=" * 70)
    print("W6D1 - LANGCHAIN CHAIN DEMO")
    print("=" * 70)

    clear_memory()

    for index, question in enumerate(
        CHAIN_QUESTIONS,
        start=1,
    ):

        print(f"\n--- Chain Test {index}/5 ---")
        print(f"Input: {question}")

        start_time = time.time()

        with mlflow.start_run(
            run_name=f"chain-test-{index}"
        ):

            mlflow.log_param(
                "component",
                "langchain_chain",
            )

            mlflow.log_param(
                "model",
                "llama3.2:3b",
            )

            mlflow.log_param(
                "input",
                question,
            )

            response = run_chain(question)

            duration = time.time() - start_time

            mlflow.log_metric(
                "latency_seconds",
                duration,
            )

            mlflow.log_text(
                json.dumps(
                    response,
                    indent=2,
                ),
                "response.json",
            )

        print("Output:")
        print(
            json.dumps(
                response,
                indent=2,
            )
        )


def run_memory_demo():

    print("\n")
    print("=" * 70)
    print("CONVERSATION BUFFER MEMORY - 5 TURNS")
    print("=" * 70)

    history = get_conversation_history()

    print("\nConversation history maintained:")
    print(history)


def run_agent_demo():

    print("\n")
    print("=" * 70)
    print("W6D1 - LANGCHAIN AGENT DEMO")
    print("=" * 70)

    for index, task in enumerate(
        AGENT_TASKS,
        start=1,
    ):

        print(f"\n--- Agent Task {index}/3 ---")
        print(f"Input: {task}")

        start_time = time.time()

        with mlflow.start_run(
            run_name=f"agent-task-{index}"
        ):

            mlflow.log_param(
                "component",
                "langchain_agent",
            )

            mlflow.log_param(
                "model",
                "llama3.2:3b",
            )

            mlflow.log_param(
                "max_iterations",
                3,
            )

            mlflow.log_param(
                "input",
                task,
            )

            response = run_agent(task)

            duration = time.time() - start_time

            mlflow.log_metric(
                "latency_seconds",
                duration,
            )

            mlflow.log_text(
                json.dumps(
                    response,
                    indent=2,
                ),
                "agent_response.json",
            )

        print("Output:")
        print(
            json.dumps(
                response,
                indent=2,
            )
        )


def main():

    run_chain_demo()

    run_memory_demo()

    run_agent_demo()

    print("\n")
    print("=" * 70)
    print("W6D1 DEMO COMPLETED")
    print("=" * 70)
    print("\nMLflow tracking database:", MLFLOW_DB)


if __name__ == "__main__":
    main()