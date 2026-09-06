"""
Command-line application for the Local AI Research Assistant.

Run with:

    python -m src.main
"""

import mlflow

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LLM_MODEL,
    MLFLOW_EXPERIMENT,
    TOP_K,
)
from src.graph import run_research


def setup_mlflow() -> None:
    """
    Configure the local MLflow experiment.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def run_application() -> None:
    """
    Start the interactive research assistant.
    """

    setup_mlflow()

    print("\n" + "=" * 60)
    print("       LOCAL AI RESEARCH ASSISTANT")
    print("=" * 60)

    print("\nType 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Research question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a research question.\n")
            continue

        try:
            with mlflow.start_run(run_name="research_query"):

                # Track configuration used for this run.
                mlflow.log_params(
                    {
                        "llm_model": LLM_MODEL,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                        "top_k": TOP_K,
                    }
                )

                result = run_research(question)

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                documents = result.get("documents", [])

                # Track simple retrieval information.
                mlflow.log_metric(
                    "retrieved_documents",
                    len(documents),
                )

                # Save the answer as an MLflow artifact.
                mlflow.log_text(
                    answer,
                    "answer.txt",
                )

            print("\n" + "-" * 60)
            print("ANSWER")
            print("-" * 60)
            print(answer)

            print("\nSources:")
            for index, document in enumerate(documents, start=1):
                source = document.metadata.get(
                    "source",
                    "Unknown",
                )

                page = document.metadata.get(
                    "page",
                    "Unknown",
                )

                print(
                    f"{index}. {source} | page {page}"
                )

            print("-" * 60 + "\n")

        except Exception as exc:
            print(
                "\nError while processing the question:"
            )
            print(exc)
            print(
                "\nCheck that Ollama is running and "
                "the required models are installed.\n"
            )


if __name__ == "__main__":
    run_application()