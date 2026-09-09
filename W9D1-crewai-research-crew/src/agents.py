from crewai import Agent, LLM


# Local Ollama LLM
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)


def create_researcher(tools=None):
    """
    Create the Researcher agent.

    The Researcher collects useful and reliable information
    about the given topic.
    """
    return Agent(
        role="Researcher",
        goal=(
            "Research the given topic carefully and collect "
            "accurate, useful and relevant information."
        ),
        backstory=(
            "You are an experienced AI research assistant. "
            "You are curious, detail-oriented and good at "
            "finding important facts. You explain information "
            "in simple and clear language."
        ),
        llm=llm,
        tools=tools or [],
        verbose=True,
        allow_delegation=False,
    )


def create_writer():
    """
    Create the Writer agent.

    The Writer converts research findings into a clear article.
    """
    return Agent(
        role="Technical Writer",
        goal=(
            "Turn research findings into a clear, simple and "
            "well-structured article."
        ),
        backstory=(
            "You are a skilled technical writer who can explain "
            "technical topics in simple English. You organize "
            "information logically and avoid unnecessary complexity."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_reviewer():
    """
    Create the Reviewer agent.

    The Reviewer checks the final article for quality,
    accuracy and completeness.
    """
    return Agent(
        role="Quality Reviewer",
        goal=(
            "Review the written article for accuracy, clarity, "
            "completeness and logical structure."
        ),
        backstory=(
            "You are a strict but helpful content reviewer. "
            "You identify missing information, unclear statements "
            "and unsupported claims. You provide practical improvements."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )