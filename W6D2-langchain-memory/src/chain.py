from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from src.config import MODEL_NAME, TEMPERATURE


def create_chain():
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a helpful AI/ML mentor.\n"
            "Answer the question clearly and briefly.\n\n"
            "Question: {question}\n"
            "Answer:"
        ),
    )

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


def ask_question(question: str) -> str:
    chain = create_chain()
    return chain.invoke({"question": question})