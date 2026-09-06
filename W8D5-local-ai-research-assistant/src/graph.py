"""
LangGraph workflow for the Local AI Research Assistant.

Workflow:

START
  |
  v
retrieve
  |
  v
generate
  |
  v
END
"""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from src.llm import get_llm
from src.retriever import format_documents, get_retriever


class ResearchState(TypedDict, total=False):
    """
    State shared between LangGraph nodes.
    """

    question: str
    documents: list[Document]
    context: str
    answer: str


PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful local AI research assistant.

Answer the user's research question using ONLY the provided context.

If the context does not contain enough information to answer the question,
say that the available documents do not contain enough information.

Do not invent facts.

Always provide a concise and clear answer.

Context:
{context}

Research Question:
{question}

Answer:
"""
)


def build_graph(retriever=None, llm=None):
    """
    Build and compile the LangGraph research workflow.

    Args:
        retriever: Optional retriever. Useful for testing.
        llm: Optional LLM. Useful for testing.

    Returns:
        Compiled LangGraph workflow.
    """

    retriever = retriever or get_retriever()
    llm = llm or get_llm()

    def retrieve_node(state: ResearchState) -> dict:
        """
        Retrieve documents relevant to the user's question.
        """
        question = state["question"]

        documents = retriever.invoke(question)

        context = format_documents(documents)

        return {
            "documents": documents,
            "context": context,
        }

    def generate_node(state: ResearchState) -> dict:
        """
        Generate the final research answer using retrieved context.
        """
        question = state["question"]
        context = state.get("context", "")

        prompt = PROMPT.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        response = llm.invoke(prompt)

        return {
            "answer": response.content
        }

    workflow = StateGraph(ResearchState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_research(
    question: str,
    retriever=None,
    llm=None,
) -> dict:
    """
    Run the complete research workflow.

    Args:
        question: User's research question.
        retriever: Optional test retriever.
        llm: Optional test LLM.

    Returns:
        Final graph state.
    """
    if not question or not question.strip():
        raise ValueError("Research question cannot be empty.")

    graph = build_graph(
        retriever=retriever,
        llm=llm,
    )

    return graph.invoke(
        {
            "question": question.strip()
        }
    )