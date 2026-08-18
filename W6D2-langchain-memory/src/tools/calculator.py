import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        return OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        operand = _calculate_node(node.operand)
        return OPERATORS[type(node.op)](operand)

    raise ValueError("Unsupported expression")


def calc(expression: str) -> str:
    try:
        expression = expression.strip()

        # Handle expressions passed by the ReAct agent with quotes
        if (
            len(expression) >= 2
            and expression[0] in ("'", '"')
            and expression[-1] == expression[0]
        ):
            expression = expression[1:-1].strip()

        tree = ast.parse(expression, mode="eval")
        result = _calculate_node(tree.body)

        return str(result)

    except Exception as exc:
        return f"Calculation error: {exc}"