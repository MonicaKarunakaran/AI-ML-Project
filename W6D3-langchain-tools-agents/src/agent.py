from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.tools.calculator import calculator
from src.tools.web_search_stub import web_search


def create_tools():
    """Return the tools available to the agent."""

    return [calculator, web_search]


def create_langchain_agent():
    """Create a LangChain agent with calculator and web-search tools."""

    llm = ChatOllama(model="llama3.2:3b")

    tools = create_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful AI/ML assistant. "
            "Use the calculator tool for mathematical calculations. "
            "Use the web search tool when the user asks for information "
            "that would normally require a web search. "
            "Give a concise final answer."
        ),
    )

    return agent


def run_agent_task(task: str) -> str:
    """Run a task through the LangChain agent."""

    agent = create_langchain_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
    )

    messages = result.get("messages", [])

    if messages:
        return messages[-1].content

    return str(result)