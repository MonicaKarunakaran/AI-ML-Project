from typing import Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from .tools import TOOLS


MODEL_NAME = "llama3.2:3b"


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

You have access to two tools:

1. calculator
   - Use this for mathematical calculations.

2. search_stub
   - Use this when the user asks about placement statistics
     or information that requires searching the local dataset.

Always use the appropriate tool when necessary.

Return a concise final answer.
""",
        ),
        (
            "human",
            "{input}",
        ),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),
    ]
)


agent = create_tool_calling_agent(
    llm=llm,
    tools=TOOLS,
    prompt=agent_prompt,
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True,
)


def run_agent(task: str) -> Dict[str, Any]:
    """
    Execute one task using the LangChain agent.
    """

    result = agent_executor.invoke(
        {
            "input": task,
        }
    )

    return {
        "input": task,
        "output": result.get("output", ""),
    }


if __name__ == "__main__":

    tasks = [
        "Calculate 25 * 16.",
        "Search for the placement statistics of Data Science.",
        "Find the highest placement percentage and calculate what 10% of that percentage is.",
    ]

    for index, task in enumerate(tasks, start=1):

        print(f"\n=== Agent Task {index} ===")
        print(f"Task: {task}")

        result = run_agent(task)

        print(f"Result: {result}")