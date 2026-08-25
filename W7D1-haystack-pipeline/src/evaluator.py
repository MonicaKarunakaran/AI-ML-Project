import re
from pathlib import Path


def load_questions(question_file: Path):
    """
    Load questions from a text file.

    Numbering such as:
        1. What is...
        2. What is...

    is automatically removed.
    """

    if not question_file.exists():
        raise FileNotFoundError(
            f"Question file not found: {question_file}"
        )

    questions = []

    with open(
        question_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Remove question numbering.
            # Example:
            # "1. What is..." -> "What is..."
            line = re.sub(
                r"^\d+\.\s*",
                "",
                line
            )

            questions.append(line)

    return questions


def calculate_precision(
    relevant_count,
    retrieved_count
):
    """
    Calculate retrieval precision.

    Precision =
        Relevant Retrieved Documents /
        Total Retrieved Documents
    """

    if retrieved_count == 0:
        return 0.0

    return relevant_count / retrieved_count


def evaluate_question(
    question,
    bm25_documents,
    dense_documents
):
    """
    Store BM25 and Dense retrieval results
    for one question.
    """

    return {
        "question": question,
        "bm25_documents": bm25_documents,
        "dense_documents": dense_documents,
    }


def evaluate_retrieval(
    questions,
    bm25_function,
    dense_function,
    bm25_store,
    dense_store
):
    """
    Run all questions through both BM25
    and Dense retrieval.
    """

    results = []

    total_questions = len(questions)

    for index, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"\nQuestion {index}/{total_questions}"
        )

        print(
            f"Query: {question}"
        )

        # -----------------------------
        # BM25
        # -----------------------------

        bm25_documents = bm25_function(
            question,
            bm25_store
        )

        # -----------------------------
        # Dense
        # -----------------------------

        dense_documents = dense_function(
            question,
            dense_store
        )

        result = evaluate_question(
            question,
            bm25_documents,
            dense_documents
        )

        results.append(result)

    return results