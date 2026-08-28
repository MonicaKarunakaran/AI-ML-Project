from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from src.ingest import load_documents
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL


SYSTEM_PROMPT = """
You are an AI/ML tutor and a helpful RAG assistant.
Answer questions clearly and concisely using the provided context.
Focus on artificial intelligence, machine learning, NLP, and RAG concepts.
If the answer is not available in the provided context, say so instead
of making up information.
"""


def build_rag(model_name="llama3.2:3b"):
    """Build the local LlamaIndex RAG pipeline."""

    Settings.llm = Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        request_timeout=300.0,
        system_prompt=SYSTEM_PROMPT,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    documents = load_documents()

    index = VectorStoreIndex.from_documents(documents)

    return index


def create_query_engine(model_name="llama3.2:3b"):
    """Create a query engine from a single local index."""

    index = build_rag(model_name)

    return index.as_query_engine(
        similarity_top_k=3
    )


def ask_question(query_engine, question):
    """Ask a question using an existing query engine."""

    response = query_engine.query(question)

    return str(response)
