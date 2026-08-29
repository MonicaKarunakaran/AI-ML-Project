from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from src.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from src.ingestion import configure_embedding_model, configure_chunking, create_vector_index


def configure_llm():
    """Configure the local Ollama LLM."""

    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=0.1,
    )


def get_query_engine():
    """
    Create a LlamaIndex query engine using the
    multi-document vector index.
    """

    configure_embedding_model()
    configure_chunking()
    configure_llm()

    index = create_vector_index()

    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K
    )

    return query_engine


def ask_question(question):
    """
    Ask a question against all indexed documents.

    Args:
        question: User question.

    Returns:
        Response object from LlamaIndex.
    """

    query_engine = get_query_engine()

    response = query_engine.query(question)

    return response


def extract_sources(response):
    """
    Extract source information from the response.
    """

    sources = []

    for source_node in response.source_nodes:

        metadata = source_node.node.metadata or {}

        source = metadata.get(
            "file_name",
            metadata.get("file_path", "Unknown")
        )

        page = metadata.get(
            "page_label",
            metadata.get("page_number", "Unknown")
        )

        sources.append(
            {
                "source": source,
                "page": page,
                "score": source_node.score,
            }
        )

    return sources


def answer_question(question):
    """
    Generate a grounded answer and return source information.
    """

    response = ask_question(question)

    sources = extract_sources(response)

    return response.response, sources