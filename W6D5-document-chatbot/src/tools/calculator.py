import numexpr as ne


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    """

    try:
        result = ne.evaluate(expression)

        if hasattr(result, "item"):
            result = result.item()

        return str(result)

    except Exception as exc:
        return f"Calculation error: {exc}"