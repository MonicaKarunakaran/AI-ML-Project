import pytest


def test_compressor_import():
    try:
        from src.compressor import (
            PromptCompressor,
        )
    except ImportError:
        pytest.skip(
            "LLMLingua is not installed"
        )

    assert PromptCompressor is not None