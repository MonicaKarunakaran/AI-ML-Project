import pytest

from src.chains.doc_qa_chain import get_chain
from src.chains.agent import create_chatbot_agent


@pytest.fixture
def chain_and_memory():
    return get_chain()


@pytest.fixture
def agent():
    return create_chatbot_agent()