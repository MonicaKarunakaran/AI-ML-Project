from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Return a simulated web search result for a query."""

    return (
        f"Web search stub result for '{query}'. "
        "This is a simulated search tool for the W6D3 demonstration."
    )