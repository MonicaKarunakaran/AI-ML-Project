import os


def get_web_search_tool():
    """
    Create the CrewAI web search tool if a Serper API key exists.

    Returns:
        SerperDevTool or None
    """

    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        print("SERPER_API_KEY not found.")
        print("Running without web search.")
        return None

    try:
        from crewai_tools import SerperDevTool

        return SerperDevTool()

    except ImportError:
        print("crewai-tools is not installed.")
        print("Install it using: pip install 'crewai[tools]'")
        return None