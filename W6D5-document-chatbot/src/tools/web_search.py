def web_search(query: str) -> str:
    """
    Stub web-search tool.

    This does not make a real web request.
    It simulates a search result so that the
    agent can demonstrate tool routing.
    """

    return (
        f"Stub search result for '{query}': "
        "LangChain is a framework for building applications "
        "powered by language models, including chains, agents, "
        "memory, retrieval, and tool integration."
    )