from llama_index.core import Settings

from src.config import TOP_K
from src.ingestion import create_vector_index


def get_retriever():
    """
    Create a vector retriever for the multi-document index.

    Returns:
        BaseRetriever
    """

    index = create_vector_index()

    retriever = index.as_retriever(
        similarity_top_k=TOP_K
    )

    return retriever


def retrieve_documents(query):
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query: User question.

    Returns:
        list: Retrieved nodes.
    """

    retriever = get_retriever()

    nodes = retriever.retrieve(query)

    return nodes


def display_retrieval_results(query):
    """
    Display retrieved chunks and their source metadata.
    """

    nodes = retrieve_documents(query)

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuestion: {query}")

    if not nodes:
        print("\nNo relevant documents found.")
        return nodes

    for index, node in enumerate(nodes, start=1):

        metadata = node.node.metadata or {}

        source = metadata.get(
            "file_name",
            metadata.get("file_path", "Unknown")
        )

        page = metadata.get(
            "page_label",
            metadata.get("page_number", "Unknown")
        )

        print(f"\n--- Result {index} ---")
        print(f"Source: {source}")
        print(f"Page: {page}")
        print(f"Score: {node.score}")
        print(f"Text: {node.node.get_content()[:500]}")

    return nodes