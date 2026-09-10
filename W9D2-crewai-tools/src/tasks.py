from crewai import Task

from src.agents import researcher, writer, reviewer


research_task = Task(
    description=(
        "Research the topic: {topic}. "
        "Use the web search tool to find recent and reliable information. "
        "Identify important facts, applications, benefits, challenges "
        "and recent developments. "
        "Use the code execution tool if a simple calculation is useful. "
        "Include source URLs wherever possible."
    ),
    expected_output=(
        "A detailed research summary containing key facts, recent "
        "developments, applications, benefits, challenges and source URLs."
    ),
    agent=researcher,
)


writing_task = Task(
    description=(
        "Using the researcher's findings, write a well-structured "
        "research report about {topic}. "
        "Use clear headings and simple technical language. "
        "Include an introduction, key developments, applications, "
        "advantages, challenges and conclusion."
    ),
    expected_output=(
        "A clear and professional research report with proper headings "
        "and logically organized information."
    ),
    agent=writer,
    context=[research_task],
)


review_task = Task(
    description=(
        "Review the research report about {topic}. "
        "Check the report for factual accuracy, completeness, clarity, "
        "organization and relevance. "
        "Identify missing information and improve weak sections. "
        "Return the corrected final report."
    ),
    expected_output=(
        "A reviewed and improved final research report that is accurate, "
        "complete, clear and well structured."
    ),
    agent=reviewer,
    context=[writing_task],
)