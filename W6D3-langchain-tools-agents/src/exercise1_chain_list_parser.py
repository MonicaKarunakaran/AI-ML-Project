from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.parsers.json_list_parser import JsonListParser


def create_keyword_chain():
    """Create a chain that generates a JSON list of keywords."""

    prompt = PromptTemplate.from_template(
        """You are an AI/ML assistant.

Given the topic below, return exactly 3 relevant keywords.

Return ONLY a valid JSON list of strings.
Do not include markdown or explanations.

Topic: {topic}
"""
    )

    llm = ChatOllama(model="llama3.2:3b")

    parser = JsonListParser()

    return prompt | llm | parser


def run_keyword_chain(topic: str) -> list[str]:
    """Generate and parse keywords for a topic."""

    chain = create_keyword_chain()

    return chain.invoke({"topic": topic})


if __name__ == "__main__":
    topics = [
        "LangChain",
        "RAG",
        "Machine Learning",
        "AI Agents",
        "Prompt Engineering",
    ]

    print("=" * 60)
    print("W6D3 - EXERCISE 1: CUSTOM JSON LIST PARSER")
    print("=" * 60)

    for topic in topics:
        print(f"\nTopic: {topic}")

        try:
            result = run_keyword_chain(topic)
            print(f"Parsed keywords: {result}")
        except Exception as exc:
            print(f"Error: {exc}")