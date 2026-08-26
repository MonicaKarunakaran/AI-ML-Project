from src.chain import create_chain


def test_chain_creation():
    chain = create_chain()
    assert chain is not None


def test_chain_has_expected_components():
    chain = create_chain()
    assert chain is not None