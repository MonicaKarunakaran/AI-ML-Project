"""
Tests for TF-IDF feature engineering.
"""

from src.feature_engineering import (
    create_tfidf_vectorizer,
    fit_transform_text,
    transform_text,
)


def test_create_vectorizer():

    vectorizer = create_tfidf_vectorizer()

    assert vectorizer is not None


def test_fit_transform():

    texts = [
        "I love this product",
        "This product is terrible",
        "Amazing experience",
        "Very bad experience",
    ]

    vectorizer = create_tfidf_vectorizer()

    features = fit_transform_text(
        vectorizer,
        texts,
    )

    assert features.shape[0] == 4
    assert features.shape[1] > 0


def test_transform():

    train_texts = [
        "good movie",
        "bad movie",
    ]

    test_texts = [
        "good experience",
    ]

    vectorizer = create_tfidf_vectorizer()

    fit_transform_text(
        vectorizer,
        train_texts,
    )

    test_features = transform_text(
        vectorizer,
        test_texts,
    )

    assert test_features.shape[0] == 1