"""
LangGraph workflow for sentiment prediction.
"""

from pathlib import Path
from typing import TypedDict

import joblib

from langgraph.graph import END, START, StateGraph


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


class SentimentState(TypedDict, total=False):
    text: str
    sentiment: str
    confidence: float
    error: str


def load_artifacts():
    """
    Load the trained model and TF-IDF vectorizer.
    """

    model_path = OUTPUT_DIR / "best_model.joblib"
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Best model not found. Run the training pipeline first."
        )

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run the training pipeline first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


def predict_sentiment(
    state: SentimentState,
):
    """
    Predict sentiment for input text.
    """

    text = state.get("text", "").strip()

    if not text:
        return {
            "error": "Text cannot be empty."
        }

    model, vectorizer = load_artifacts()

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    confidence = float(max(probability))

    sentiment = (
        "positive"
        if prediction == 1
        else "negative"
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
    }


def build_sentiment_graph():
    """
    Build the LangGraph sentiment prediction workflow.
    """

    graph = StateGraph(SentimentState)

    graph.add_node(
        "predict_sentiment",
        predict_sentiment,
    )

    graph.add_edge(
        START,
        "predict_sentiment",
    )

    graph.add_edge(
        "predict_sentiment",
        END,
    )

    return graph.compile()


def predict(text: str):
    """
    Public prediction function.
    """

    graph = build_sentiment_graph()

    result = graph.invoke(
        {
            "text": text
        }
    )

    return result


if __name__ == "__main__":

    result = predict(
        "I absolutely loved this movie!"
    )

    print(result)