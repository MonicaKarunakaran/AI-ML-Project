from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


def create_chain():
    """Create a basic LangChain prompt → Ollama → parser chain."""

    prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a knowledgeable AI/ML learning assistant.\n"
        "Give accurate, concise answers using standard AI/ML terminology.\n"
        "RAG means Retrieval-Augmented Generation.\n"
        "LangChain is a framework for developing applications powered by "
        "language models, including chains, agents, tools, and retrieval.\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)

    llm = OllamaLLM(model="llama3.2:3b")

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain


def ask_question(question: str) -> str:
    """Run a question through the LangChain chain."""

    chain = create_chain()
    return chain.invoke({"question": question})