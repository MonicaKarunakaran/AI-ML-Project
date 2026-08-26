from src.llm_client import MockLLMClient


def test_mock_llm_client():
    client = MockLLMClient()

    response, input_tokens, output_tokens = (
        client.generate(
            "What is artificial intelligence?"
        )
    )

    assert response
    assert input_tokens > 0
    assert output_tokens > 0