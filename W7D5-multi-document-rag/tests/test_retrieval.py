from src.ingestion import create_vector_index
from src.retrieval import retrieve_documents


def test_create_vector_index():
    """Verify that the multi-document vector index can be created."""

    index = create_vector_index()

    assert index is not None


def test_retrieve_relevant_documents():
    """Verify that relevant document chunks can be retrieved."""

    nodes = retrieve_documents(
        "What is Newton's second law?"
    )

    assert nodes
    assert len(nodes) > 0


def test_retrieved_nodes_have_content():
    """Verify that retrieved nodes contain text."""

    nodes = retrieve_documents(
        "What is photosynthesis?"
    )

    assert nodes

    for node in nodes:
        assert node.node.get_content()