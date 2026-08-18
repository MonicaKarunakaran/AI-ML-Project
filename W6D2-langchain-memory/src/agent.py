from langchain.agents import AgentType, initialize_agent
from langchain_ollama import OllamaLLM

from src.config import MODEL_NAME, TEMPERATURE
from src.tools import TOOLS


def create_agent():
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="force",
    )

    return agent


def run_agent_task(task: str) -> str:
    agent = create_agent()
    return agent.run(task)