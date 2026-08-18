import json
from typing import Any, Dict

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2:3b"


# Local Ollama model
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


# Parser converts the LLM response into Python JSON/dict
output_parser = JsonOutputParser()


# PromptTemplate
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a helpful AI/ML assistant.

Use accurate and standard AI/ML terminology.

Important terminology:
- RAG means Retrieval-Augmented Generation.
- LLM means Large Language Model.
- Machine learning is a subset of artificial intelligence.
- Deep learning is a subset of machine learning.
- Deep learning uses multi-layer neural networks.
- LangChain is a framework for building applications powered by language
  models using prompts, models, parsers, tools, and agents.

If the question asks about one of these terms, use the definitions above.

Conversation history:
{history}

Current question:
{question}

Return ONLY valid JSON using exactly this format:

{{
    "question": "the user's question",
    "answer": "a clear and accurate answer"
}}

Do not use markdown.
Do not add text outside the JSON object.
"""
)

# Prompt → Ollama → OutputParser
chain = prompt | llm | output_parser


# ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    output_key="response",
    return_messages=False,
)


def run_chain(question: str) -> Dict[str, Any]:
    """
    Run one question through the LangChain pipeline
    and store the interaction in ConversationBufferMemory.
    """

    history = memory.load_memory_variables({}).get("history", "")

    try:
        parsed_response = chain.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        # Make sure the result is a dictionary
        if not isinstance(parsed_response, dict):
            parsed_response = {
                "question": question,
                "answer": str(parsed_response),
            }

    except Exception:
        # Fallback in case the local model returns malformed JSON
        raw_response = (
            prompt
            | llm
        ).invoke(
            {
                "history": history,
                "question": question,
            }
        )

        parsed_response = {
            "question": question,
            "answer": raw_response.content,
        }

    # Save conversation turn
    memory.save_context(
        {"question": question},
        {"response": json.dumps(parsed_response)},
    )

    return parsed_response


def get_conversation_history() -> str:
    """Return the current conversation history."""

    return memory.load_memory_variables({}).get("history", "")


def clear_memory() -> None:
    """Clear the conversation history."""

    memory.clear()


if __name__ == "__main__":

    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is RAG?",
        "What is an LLM?",
        "What is LangChain?",
    ]

    for index, question in enumerate(questions, start=1):

        print(f"\n--- Turn {index} ---")
        print(f"Question: {question}")

        response = run_chain(question)

        print("Response:")
        print(json.dumps(response, indent=2))

    print("\n=== Conversation History ===")
    print(get_conversation_history())