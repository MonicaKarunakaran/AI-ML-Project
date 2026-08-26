from pathlib import Path

from src.config import (
    PDF_DIR,
    QUESTIONS_FILE,
    OUTPUT_DIR,
)

from src.document_loader import load_documents
from src.pipeline_bm25 import run_bm25
from src.pipeline_dense import (
    prepare_embeddings,
    run_dense,
)

from src.evaluator import load_questions


EXPECTED_SOURCES = {
    "physics": "135742-NEET Physics_Paper_With_Answer-pdf.pdf",
    "chemistry": "1407-Chemistry_Paper_With_Answer-NEET-pdf.pdf",
    "biology": "135925-NEET-2024-bio-question-paper-pdf.pdf",
    "gita": "Bhagavad Gita-As It Is.pdf",
    "novel": "We Were Never Meant to Be.pdf",
}


def get_expected_source(question_number):
    """Return expected PDF for a question."""

    if 1 <= question_number <= 10:
        return EXPECTED_SOURCES["physics"]

    if 11 <= question_number <= 20:
        return EXPECTED_SOURCES["chemistry"]

    if 21 <= question_number <= 30:
        return EXPECTED_SOURCES["biology"]

    if 31 <= question_number <= 40:
        return EXPECTED_SOURCES["gita"]

    if 41 <= question_number <= 50:
        return EXPECTED_SOURCES["novel"]

    return None


def get_source(document):
    """Extract source filename from document metadata."""

    source = document.meta.get("file_path")

    if source is None:
        source = document.meta.get("source")

    if source is None:
        source = document.meta.get("file_name")

    if source is None:
        return "Unknown"

    return Path(str(source)).name


def calculate_precision(documents, expected_source, k=5):
    """Calculate Precision@K based on source PDF."""

    top_documents = documents[:k]

    if not top_documents:
        return 0.0, 0

    relevant = 0

    for document in top_documents:

        source = get_source(document)

        if source == expected_source:
            relevant += 1

    precision = relevant / len(top_documents)

    return precision, relevant


def save_retrieval_results(
    filename,
    results,
    method,
):
    """Save detailed retrieval results."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / filename

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"{method} RETRIEVAL RESULTS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for (
            question_number,
            question,
            documents,
        ) in results:

            expected_source = get_expected_source(
                question_number
            )

            file.write(
                f"Question {question_number}: "
                f"{question}\n"
            )

            file.write(
                f"Expected Source: "
                f"{expected_source}\n"
            )

            file.write(
                "-" * 70 + "\n"
            )

            for rank, document in enumerate(
                documents[:5],
                start=1,
            ):

                source = get_source(document)

                score = document.score

                file.write(
                    f"\nRank {rank}\n"
                )

                file.write(
                    f"Source: {source}\n"
                )

                if score is not None:
                    file.write(
                        f"Score: {score:.4f}\n"
                    )

                file.write(
                    "\n"
                )

                file.write(
                    document.content[:800]
                )

                file.write(
                    "\n\n"
                )


def save_comparison(
    bm25_results,
    dense_results,
):
    """Calculate and save BM25 vs Dense comparison."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_file = (
        OUTPUT_DIR / "comparison.txt"
    )

    bm25_precisions = []
    dense_precisions = []

    with open(
        comparison_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "BM25 vs DENSE RETRIEVAL COMPARISON\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for index in range(
            len(bm25_results)
        ):

            question_number = (
                bm25_results[index][0]
            )

            question = (
                bm25_results[index][1]
            )

            bm25_documents = (
                bm25_results[index][2]
            )

            dense_documents = (
                dense_results[index][2]
            )

            expected_source = (
                get_expected_source(
                    question_number
                )
            )

            bm25_precision, bm25_relevant = (
                calculate_precision(
                    bm25_documents,
                    expected_source,
                    k=5,
                )
            )

            dense_precision, dense_relevant = (
                calculate_precision(
                    dense_documents,
                    expected_source,
                    k=5,
                )
            )

            bm25_precisions.append(
                bm25_precision
            )

            dense_precisions.append(
                dense_precision
            )

            file.write(
                f"Question {question_number}: "
                f"{question}\n"
            )

            file.write(
                f"Expected Source: "
                f"{expected_source}\n\n"
            )

            file.write(
                f"BM25 Precision@5: "
                f"{bm25_precision:.2f} "
                f"({bm25_relevant}/5)\n"
            )

            file.write(
                f"Dense Precision@5: "
                f"{dense_precision:.2f} "
                f"({dense_relevant}/5)\n"
            )

            if bm25_precision > dense_precision:
                winner = "BM25"

            elif dense_precision > bm25_precision:
                winner = "Dense"

            else:
                winner = "Tie"

            file.write(
                f"Question Winner: {winner}\n"
            )

            file.write(
                "-" * 70 + "\n\n"
            )

        # Overall averages

        bm25_average = (
            sum(bm25_precisions)
            / len(bm25_precisions)
        )

        dense_average = (
            sum(dense_precisions)
            / len(dense_precisions)
        )

        file.write(
            "\nOVERALL RESULTS\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            f"Questions Evaluated: "
            f"{len(bm25_results)}\n"
        )

        file.write(
            f"BM25 Average Precision@5: "
            f"{bm25_average:.2%}\n"
        )

        file.write(
            f"Dense Average Precision@5: "
            f"{dense_average:.2%}\n"
        )

        if bm25_average > dense_average:
            overall_winner = "BM25"

        elif dense_average > bm25_average:
            overall_winner = "Dense"

        else:
            overall_winner = "Tie"

        file.write(
            f"Overall Better Retriever: "
            f"{overall_winner}\n"
        )

    print(
        f"\nComparison saved to: "
        f"{comparison_file}"
    )


def main():

    print("=" * 70)
    print("BM25 vs DENSE RETRIEVAL EVALUATION")
    print("=" * 70)

    questions = load_questions(
        QUESTIONS_FILE
    )

    print(
        f"\nLoaded {len(questions)} questions."
    )

    if len(questions) < 10:
        raise ValueError(
            "At least 10 questions are required."
        )

    # -------------------------------
    # BM25
    # -------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPARING BM25 DOCUMENT STORE"
    )

    print(
        "=" * 70
    )

    bm25_store = load_documents(
        PDF_DIR
    )

    bm25_results = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"BM25 - Question "
            f"{index}/{len(questions)}"
        )

        documents = run_bm25(
            question,
            bm25_store,
        )

        bm25_results.append(
            (
                index,
                question,
                documents,
            )
        )

    # -------------------------------
    # Dense
    # -------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPARING DENSE DOCUMENT STORE"
    )

    print(
        "=" * 70
    )

    dense_store = load_documents(
        PDF_DIR
    )

    dense_store = prepare_embeddings(
        dense_store
    )

    dense_results = []

    for index, question in enumerate(
        questions,
        start=1,
    ):

        print(
            f"Dense - Question "
            f"{index}/{len(questions)}"
        )

        documents = run_dense(
            question,
            dense_store,
        )

        dense_results.append(
            (
                index,
                question,
                documents,
            )
        )

    # -------------------------------
    # Save outputs
    # -------------------------------

    save_retrieval_results(
        "bm25_results.txt",
        bm25_results,
        "BM25",
    )

    save_retrieval_results(
        "dense_results.txt",
        dense_results,
        "DENSE",
    )

    save_comparison(
        bm25_results,
        dense_results,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()