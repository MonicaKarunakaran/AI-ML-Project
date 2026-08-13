import mlflow

from data_prep import load_data
from tuning import grid_search_svm
from tuning import random_search_knn
from utils import evaluate
from crewai_agent import review_model


def run():

    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("W3D5 Hyperparameter Tuning")

    ###########################
    # Grid Search
    ###########################

    with mlflow.start_run(run_name="GridSearch_SVM"):

        grid = grid_search_svm(X_train, y_train)

        accuracy = evaluate(grid.best_estimator_, X_test, y_test)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)

        review_model(
            "GridSearch SVM",
            grid.best_params_,
            grid.best_score_
        )

        with open("outputs/results.txt", "a") as file:
            file.write("GRID SEARCH\n")
            file.write(str(grid.best_params_) + "\n")
            file.write(f"{accuracy}\n\n")

    ###########################
    # Random Search
    ###########################

    with mlflow.start_run(run_name="RandomSearch_KNN"):

        random = random_search_knn(X_train, y_train)

        accuracy = evaluate(random.best_estimator_, X_test, y_test)

        mlflow.log_params(random.best_params_)
        mlflow.log_metric("accuracy", accuracy)

        review_model(
            "Random Search KNN",
            random.best_params_,
            random.best_score_
        )

        with open("outputs/results.txt", "a") as file:
            file.write("RANDOM SEARCH\n")
            file.write(str(random.best_params_) + "\n")
            file.write(f"{accuracy}\n\n")


if __name__ == "__main__":
    run()