from crewai.tools import tool
from ddgs import DDGS
import ast
import operator


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web and return recent information about a topic."""

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No web search results were found."

        output = []

        for result in results:
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "No URL")

            output.append(
                f"Title: {title}\n"
                f"Information: {body}\n"
                f"Source: {url}\n"
            )

        return "\n---\n".join(output)

    except Exception as e:
        return f"Web search failed: {e}"


# Safe arithmetic code execution
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp):
        operator_function = _ALLOWED_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Operator not allowed")

        return operator_function(
            _evaluate(node.left),
            _evaluate(node.right)
        )

    raise ValueError("Only basic arithmetic expressions are allowed")


@tool("code_execution")
def code_execution(expression: str) -> str:
    """Execute a basic mathematical expression and return the result."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)

        return f"Expression: {expression}\nResult: {result}"

    except Exception as e:
        return f"Code execution failed: {e}"