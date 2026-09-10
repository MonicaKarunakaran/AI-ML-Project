from crewai import Agent, LLM

from src.tools import web_search, code_execution


llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)


researcher = Agent(
    role="AI Researcher",
    goal=(
        "Research the assigned topic using web search and identify "
        "accurate, recent and relevant information."
    ),
    backstory=(
        "You are an experienced AI researcher. You collect reliable "
        "information from multiple sources and identify important facts. "
        "You can also use code execution for simple calculations."
    ),
    tools=[web_search, code_execution],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


writer = Agent(
    role="Technical Writer",
    goal=(
        "Transform the research findings into a clear, structured and "
        "easy-to-understand technical report."
    ),
    backstory=(
        "You are an experienced technical writer who explains complex "
        "AI and ML topics in simple and organized language."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


reviewer = Agent(
    role="Research Reviewer",
    goal=(
        "Review the final report for accuracy, completeness, clarity, "
        "structure and relevance."
    ),
    backstory=(
        "You are a quality reviewer who carefully checks technical "
        "reports and identifies missing information, unclear statements "
        "and possible factual problems."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)