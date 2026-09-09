from crewai import Task


def create_research_task(researcher):
    """
    Task for the Researcher agent.
    """
    return Task(
        description="""
        Research the following topic:

        {topic}

        Provide useful information that can be used by a technical writer.

        Cover:
        1. What the topic means
        2. Important concepts
        3. Real-world applications
        4. Benefits
        5. Limitations or challenges
        6. Recent developments if known

        Keep the information factual and easy to understand.
        """,
        expected_output="""
        A structured research report containing:
        - Topic overview
        - Key concepts
        - Applications
        - Benefits
        - Challenges
        - Important facts
        """,
        agent=researcher,
    )


def create_writing_task(writer):
    """
    Task for the Writer agent.
    """
    return Task(
        description="""
        Using the research provided by the Researcher, write a
        clear and informative article about:

        {topic}

        The article should:
        1. Have a clear introduction
        2. Explain the main concepts
        3. Include practical applications
        4. Explain benefits and challenges
        5. Use simple English
        6. Avoid unsupported claims
        7. Be well structured with headings

        Do not mention that you are an AI agent.
        """,
        expected_output="""
        A well-structured article of approximately 500-700 words
        with headings, clear explanations, applications, benefits
        and challenges.
        """,
        agent=writer,
    )


def create_review_task(reviewer):
    """
    Task for the Reviewer agent.
    """
    return Task(
        description="""
        Review the article created by the Writer about:

        {topic}

        Check the article for:

        1. Factual accuracy
        2. Completeness
        3. Clarity
        4. Logical structure
        5. Grammar
        6. Repetition
        7. Unsupported claims

        Then provide an improved final version of the article.

        Keep the language simple and professional.
        """,
        expected_output="""
        A review containing:
        - Overall quality assessment
        - Important issues found
        - Suggested improvements
        - Final improved article
        """,
        agent=reviewer,
    )