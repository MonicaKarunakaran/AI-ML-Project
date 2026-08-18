from langchain_core.tools import StructuredTool

from src.tools.calculator import calc
from src.tools.web_search import web_search


calculator_tool = StructuredTool.from_function(
    func=calc,
    name="calculator",
    description=(
        "Use this tool to perform basic arithmetic calculations. "
        "Input should be a mathematical expression such as '25 * 4'."
    ),
)


web_search_tool = StructuredTool.from_function(
    func=web_search,
    name="web_search",
    description=(
        "Use this tool when the user asks to search for information "
        "on the web. This is a simulated web search stub."
    ),
)


TOOLS = [
    web_search_tool,
    calculator_tool,
]