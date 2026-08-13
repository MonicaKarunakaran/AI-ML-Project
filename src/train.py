"""
Training pipeline
Precision Recall AUC Evaluation
"""

import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score

from src.data import load_data
from src.evaluation import (
    calculate_metrics,
    plot_roc_curve
)

from src.mlflow_utils import (
    setup_mlflow,
    log_results
)



def main():

    setup_mlflow()


    X_train, X_test, y_train, y_test = load_data()


    model = LogisticRegression(
        max_iter=1000
    )


    with mlflow.start_run():


        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        probabilities = model.predict_proba(
            X_test
        )[:,1]


        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )


        print("\nModel Metrics")

        for k,v in metrics.items():
            print(
                f"{k}: {v:.4f}"
            )


        plot_roc_curve(
            y_test,
            probabilities
        )


        log_results(
            model,
            metrics
        )


        # Stratified Cross Validation

        cv = StratifiedKFold(
            n_splits=5
        )


        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        )


        mlflow.log_metric(
            "cv_auc_mean",
            scores.mean()
        )


        print(
            "CV AUC:",
            scores.mean()
        )



if __name__ == "__main__":
    main()