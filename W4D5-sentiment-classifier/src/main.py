"""
End-to-end sentiment classification pipeline.

Workflow:
1. Load data
2. Split data
3. Create TF-IDF features
4. Train Logistic Regression
5. Train Random Forest
6. Evaluate both models
7. Compare results
8. Log results to MLflow
9. Save the best model
"""

from pathlib import Path

import joblib
import mlflow

from sklearn.model_selection import train_test_split

from src.data_loader import (
    get_features_and_target,
    load_sentiment_data,
)

from src.feature_engineering import (
    create_tfidf_vectorizer,
    fit_transform_text,
    transform_text,
)

from src.mlflow_utils import (
    log_model_metrics,
    log_model_parameters,
    setup_mlflow,
)

from src.models.logistic_regression import (
    evaluate_logistic_regression,
    train_logistic_regression,
)

from src.models.random_forest import (
    evaluate_random_forest,
    train_random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def compare_models(
    logistic_metrics: dict,
    random_forest_metrics: dict,
):
    """
    Compare Logistic Regression and Random Forest.
    """

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"{'Metric':<15}"
        f"{'Logistic Regression':<22}"
        f"{'Random Forest':<20}"
    )

    print("-" * 70)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]:

        print(
            f"{metric:<15}"
            f"{logistic_metrics[metric]:<22.4f}"
            f"{random_forest_metrics[metric]:<20.4f}"
        )

    # Select model based on F1 score
    if (
        logistic_metrics["f1_score"]
        >= random_forest_metrics["f1_score"]
    ):
        best_model_name = "Logistic Regression"
    else:
        best_model_name = "Random Forest"

    print("\nBest model based on F1-score:")
    print(best_model_name)

    return best_model_name


def main():

    print("=" * 70)
    print("W4D5 SENTIMENT CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------
    # 1. MLflow setup
    # --------------------------------------------------

    tracking_uri = setup_mlflow()

    print(f"\nMLflow tracking URI: {tracking_uri}")

    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    df = load_sentiment_data()

    print(f"Dataset shape: {df.shape}")

    print("\nClass distribution:")
    print(df["sentiment"].value_counts())

    X, y = get_features_and_target(df)

    # --------------------------------------------------
    # 3. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --------------------------------------------------
    # 4. TF-IDF
    # --------------------------------------------------

    vectorizer = create_tfidf_vectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    X_train_tfidf = fit_transform_text(
        vectorizer,
        X_train,
    )

    X_test_tfidf = transform_text(
        vectorizer,
        X_test,
    )

    print(
        f"\nTF-IDF feature shape: "
        f"{X_train_tfidf.shape}"
    )

    # --------------------------------------------------
    # 5. Logistic Regression
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="Logistic-Regression"
    ):

        logistic_model = train_logistic_regression(
            X_train_tfidf,
            y_train,
        )

        logistic_metrics = evaluate_logistic_regression(
            logistic_model,
            X_test_tfidf,
            y_test,
        )

        log_model_parameters(
            {
                "model": "LogisticRegression",
                "C": 1.0,
                "max_iter": 1000,
            }
        )

        log_model_metrics(
            logistic_metrics
        )

        mlflow.sklearn.log_model(
            logistic_model,
            "logistic_regression_model",
        )

    # --------------------------------------------------
    # 6. Random Forest
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="Random-Forest"
    ):

        random_forest_model = train_random_forest(
            X_train_tfidf,
            y_train,
            n_estimators=200,
        )

        random_forest_metrics = evaluate_random_forest(
            random_forest_model,
            X_test_tfidf,
            y_test,
        )

        log_model_parameters(
            {
                "model": "RandomForestClassifier",
                "n_estimators": 200,
                "random_state": 42,
            }
        )

        log_model_metrics(
            random_forest_metrics
        )

        mlflow.sklearn.log_model(
            random_forest_model,
            "random_forest_model",
        )

    # --------------------------------------------------
    # 7. Compare models
    # --------------------------------------------------

    best_model_name = compare_models(
        logistic_metrics,
        random_forest_metrics,
    )

    # --------------------------------------------------
    # 8. Save best model
    # --------------------------------------------------

    if best_model_name == "Logistic Regression":

        best_model = logistic_model
        best_metrics = logistic_metrics

    else:

        best_model = random_forest_model
        best_metrics = random_forest_metrics

    model_path = OUTPUT_DIR / "best_model.joblib"

    vectorizer_path = (
        OUTPUT_DIR / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    print("\nSaved artifacts:")

    print(f"Model: {model_path}")
    print(f"Vectorizer: {vectorizer_path}")

    # --------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Best Model: {best_model_name}")

    for metric, value in best_metrics.items():

        print(
            f"{metric}: {value:.4f}"
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()