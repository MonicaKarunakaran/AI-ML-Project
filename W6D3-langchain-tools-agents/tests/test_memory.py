from src.memory import create_memory_chain


def test_memory_chain_creation():
    chain = create_memory_chain()
    assert chain is not None