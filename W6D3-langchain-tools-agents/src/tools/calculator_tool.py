from langchain_core.tools import tool


@tool
def calculator_tool(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        allowed_chars = "0123456789+-*/().% "

        if not all(char in allowed_chars for char in expression):
            return "Error: unsupported characters in expression."

        result = eval(expression, {"__builtins__": {}}, {})

        return str(result)

    except Exception as exc:
        return f"Error calculating expression: {exc}"