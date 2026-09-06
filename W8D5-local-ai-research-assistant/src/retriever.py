"""
Document loading and vector retrieval.

This module is responsible for:
1. Loading PDF documents.
2. Splitting documents into chunks.
3. Creating embeddings using Ollama.
4. Storing embeddings in ChromaDB.
5. Returning a retriever for semantic search.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
    VECTOR_DB_DIR,
)


def load_documents() -> list[Document]:
    """
    Load all PDF documents from the documents directory.

    Returns:
        list[Document]: Loaded PDF pages.

    Raises:
        FileNotFoundError: If no PDF documents are found.
    """
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in: {DOCUMENTS_DIR}"
        )

    documents: list[Document] = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for retrieval.

    Args:
        documents: Documents loaded from PDF files.

    Returns:
        list[Document]: Chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def get_embeddings() -> OllamaEmbeddings:
    """
    Create the local Ollama embedding model.

    Returns:
        OllamaEmbeddings: Local embedding model.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def build_vector_store() -> Chroma:
    """
    Build a Chroma vector store from the local documents.

    Returns:
        Chroma: Persistent Chroma vector store.
    """
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name="research_documents",
    )

    return vector_store


def get_retriever():
    """
    Create a semantic retriever.

    The vector database is created from the local documents.
    For this small capstone, rebuilding the database on startup
    keeps the implementation simple and predictable.

    Returns:
        Retriever: Chroma retriever.
    """
    vector_store = build_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def format_documents(documents: list[Document]) -> str:
    """
    Convert retrieved documents into readable context.

    Args:
        documents: Retrieved documents.

    Returns:
        str: Formatted context for the LLM.
    """
    formatted_chunks = []

    for index, document in enumerate(documents, start=1):
        source = Path(
            document.metadata.get("source", "Unknown source")
        ).name

        page = document.metadata.get("page", "Unknown")

        formatted_chunks.append(
            f"[Source {index}: {source}, page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_chunks)