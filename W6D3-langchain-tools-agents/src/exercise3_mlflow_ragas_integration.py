import json
import time
from pathlib import Path

import mlflow

from src.chain import create_chain
from src.exercise2_agent_three_tools import create_three_tool_agent


BASE_DIR = Path(__file__).resolve().parent.parent
GROUND_TRUTH_FILE = BASE_DIR / "data" / "ground_truth.json"

CHAIN_QUESTIONS = [
    "What is LangChain?",
    "What is RAG?",
    "What is an LLM?",
    "What is prompt engineering?",
    "What is an AI agent?",
]

AGENT_TASKS = [
    "Calculate 125 * 24.",
    "What holiday is on 2026-08-15?",
    "Search for information about Retrieval-Augmented Generation.",
]


def load_ground_truth():
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def run_chain_evaluation():
    chain = create_chain()

    results = []

    for question in CHAIN_QUESTIONS:
        start = time.perf_counter()

        with mlflow.start_run(run_name="chain-run", nested=True):
            output = chain.invoke({"question": question})

            latency = time.perf_counter() - start

            mlflow.log_param("component", "langchain_chain")
            mlflow.log_param("question", question)
            mlflow.log_text(str(output), "llm_output.txt")
            mlflow.log_metric("latency_seconds", latency)

        results.append(
            {
                "question": question,
                "output": str(output),
                "latency": latency,
            }
        )

    return results


def run_agent_evaluation():
    agent = create_three_tool_agent()

    results = []

    for task in AGENT_TASKS:
        start = time.perf_counter()

        with mlflow.start_run(run_name="agent-run", nested=True):
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

            latency = time.perf_counter() - start

            messages = result.get("messages", [])

            final_output = (
                messages[-1].content
                if messages
                else "No output returned."
            )

            mlflow.log_param("component", "langchain_agent")
            mlflow.log_param("task", task)
            mlflow.log_text(str(final_output), "agent_output.txt")
            mlflow.log_metric("latency_seconds", latency)

        results.append(
            {
                "task": task,
                "output": str(final_output),
                "latency": latency,
            }
        )

    return results


def run_evaluation():
    print("=" * 60)
    print("W6D3 - EXERCISE 3: MLFLOW + RAGAS INTEGRATION")
    print("=" * 60)

    ground_truth = load_ground_truth()

    print(f"\nLoaded {len(ground_truth)} ground-truth examples.")

    mlflow.set_experiment("W6D3-LangChain-Evaluation")

    with mlflow.start_run(run_name="W6D3-complete-evaluation"):
        mlflow.log_param("chain_inputs", len(CHAIN_QUESTIONS))
        mlflow.log_param("agent_tasks", len(AGENT_TASKS))
        mlflow.log_param("ragas_version", "0.4.3")

        print("\nRunning chain evaluation...")
        chain_results = run_chain_evaluation()

        print("Running agent evaluation...")
        agent_results = run_agent_evaluation()

        mlflow.log_metric(
            "total_chain_runs",
            len(chain_results),
        )

        mlflow.log_metric(
            "total_agent_runs",
            len(agent_results),
        )

    print("\nChain runs:", len(chain_results))
    print("Agent runs:", len(agent_results))

    print("\nRagas status:")
    try:
        import ragas

        print(f"Ragas {ragas.__version__} is installed.")
        print(
            "Ragas evaluation is currently unavailable because "
            "Ragas 0.4.3 requires a VertexAI integration that is "
            "not available in the installed langchain-community package."
        )
    except Exception as exc:
        print(f"Ragas import unavailable: {exc}")

    print("\nMLflow logging completed successfully.")


if __name__ == "__main__":
    run_evaluation()