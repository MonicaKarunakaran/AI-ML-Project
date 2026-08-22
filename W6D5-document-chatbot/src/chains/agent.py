from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from src.config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE
from src.tools.calculator import calculator
from src.tools.web_search import web_search


@tool
def calculator_tool(expression: str) -> str:
    """Perform a mathematical calculation."""
    return calculator(expression)


@tool
def web_search_tool(query: str) -> str:
    """Search the web using the demo search stub."""
    return web_search(query)


def create_chatbot_agent():
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

    tools = [
        calculator_tool,
        web_search_tool,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a helpful assistant. "
            "Use the calculator tool for mathematical calculations. "
            "Use the web search tool when the user asks for information "
            "that requires a search."
        ),
    )

    return agent


def run_agent_task(agent, task: str):
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

    return result