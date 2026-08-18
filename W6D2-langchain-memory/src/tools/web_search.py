def web_search(query: str) -> str:
    query = query.strip().strip("'").strip('"')

    results = {
        "langchain": (
            "LangChain is an open-source framework for building "
            "applications and workflows powered by language models."
        ),
        "machine learning": (
            "Machine learning is a branch of AI that enables systems "
            "to learn patterns from data and make predictions."
        ),
    }

    query_lower = query.lower()

    for keyword, result in results.items():
        if keyword in query_lower:
            return f"Search result for '{query}': {result}"

    return (
        f"Search result for '{query}': "
        "This is a simulated web search result."
    )