from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

try:
    from langchain_classic.memory import ConversationBufferMemory
except ImportError:
    from langchain.memory import ConversationBufferMemory

from src.config import MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE
from src.parsers.json_parser import parse_response


PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "doc_qa_prompt.txt"
)


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def create_llm():
    return OllamaLLM(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def create_memory():
    return ConversationBufferMemory(
        memory_key="history",
        input_key="question",
        output_key="answer",
        return_messages=False,
    )


def get_chain():
    prompt = PromptTemplate(
        template=load_prompt(),
        input_variables=[
            "history",
            "question",
        ],
    )

    llm = create_llm()

    chain = prompt | llm | StrOutputParser()

    memory = create_memory()

    return chain, memory


def ask_question(chain, memory, question: str) -> dict:
    history = memory.load_memory_variables({}).get(
        "history",
        "",
    )

    response = chain.invoke(
        {
            "history": history,
            "question": question,
        }
    )

    parsed_response = parse_response(response)

    answer = parsed_response["answer"]
    sources = parsed_response["sources"]

    memory.save_context(
        {"question": question},
        {"answer": answer},
    )

    return {
        "answer": answer,
        "sources": sources,
    }