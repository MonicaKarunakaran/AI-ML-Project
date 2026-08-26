import ast
import operator
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Placement dataset
DATA_FILE = PROJECT_ROOT / "data" / "placement_stats.csv"


@tool
def search_stub(query: str) -> str:
    """
    Search the local placement statistics dataset.
    This is a web-search stub used for the W6D1 demonstration.
    """

    if not DATA_FILE.exists():
        return "Placement statistics file was not found."

    df = pd.read_csv(DATA_FILE)

    query_lower = query.lower()

    # Search department names
    matches = df[
        df["department"]
        .astype(str)
        .str.lower()
        .str.contains(query_lower, na=False)
    ]

    # If no direct department match, return useful summary
    if matches.empty:

        if "placement" in query_lower:
            best = df.sort_values(
                "placement_percentage",
                ascending=False
            ).head(3)

            return (
                "Top placement departments:\n"
                + best[
                    [
                        "department",
                        "placement_percentage",
                        "average_package_lpa",
                    ]
                ].to_string(index=False)
            )

        if "package" in query_lower or "salary" in query_lower:
            best = df.sort_values(
                "average_package_lpa",
                ascending=False
            ).head(3)

            return (
                "Departments with highest average packages:\n"
                + best[
                    [
                        "department",
                        "average_package_lpa",
                    ]
                ].to_string(index=False)
            )

        return (
            "Available departments: "
            + ", ".join(df["department"].tolist())
        )

    return matches.to_string(index=False)


# Allowed mathematical operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:

    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):

        operand = _safe_eval(node.operand)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(operand)

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expr: str) -> str:
    """
    Safely calculate a mathematical expression.
    Example: 25 * 16 or (100 + 50) / 5.
    """

    try:
        tree = ast.parse(expr, mode="eval")

        result = _safe_eval(tree.body)

        return str(result)

    except Exception as exc:
        return f"Calculation error: {exc}"


TOOLS = [
    search_stub,
    calculator,
]