def evaluate_with_ragas(
    questions,
    answers,
    contexts=None,
    ground_truths=None,
):
    """
    Optional RAGAS evaluation wrapper.

    RAGAS is only needed when retrieval/context evaluation
    is required.
    """

    try:
        from ragas import evaluate
    except ImportError:
        return {
            "status": "skipped",
            "reason": "RAGAS is not installed."
        }

    dataset = {
        "question": questions,
        "answer": answers,
    }

    if contexts:
        dataset["contexts"] = contexts

    if ground_truths:
        dataset["ground_truth"] = ground_truths

    try:
        result = evaluate(dataset)

        return {
            "status": "success",
            "result": result,
        }

    except Exception as exc:
        return {
            "status": "error",
            "reason": str(exc),
        }