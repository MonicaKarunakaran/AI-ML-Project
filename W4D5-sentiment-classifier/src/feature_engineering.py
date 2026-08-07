"""
Feature engineering using TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
    )

    return vectorizer


def fit_transform_text(
    vectorizer: TfidfVectorizer,
    X_train,
):
    """
    Fit TF-IDF on training data and transform it.
    """

    X_train_tfidf = vectorizer.fit_transform(X_train)

    return X_train_tfidf


def transform_text(
    vectorizer: TfidfVectorizer,
    X_test,
):
    """
    Transform test text using an already fitted vectorizer.
    """

    X_test_tfidf = vectorizer.transform(X_test)

    return X_test_tfidf