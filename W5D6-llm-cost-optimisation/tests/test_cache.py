import numpy as np

from src.cache import (
    SemanticCache,
    exact_cache_key,
)


def test_exact_cache_key():
    key1 = exact_cache_key(
        "What is human life?"
    )

    key2 = exact_cache_key(
        "what is human life?"
    )

    assert key1 == key2


def test_cosine_similarity():
    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [1.0, 0.0]
    )

    similarity = (
        SemanticCache.cosine_similarity(
            a,
            b,
        )
    )

    assert similarity == 1.0


def test_orthogonal_vectors():
    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0]
    )

    similarity = (
        SemanticCache.cosine_similarity(
            a,
            b,
        )
    )

    assert similarity == 0.0