"""
Main Pipeline
Decision Tree + Random Forest
"""

from src.mlflow_logger import setup_mlflow

from src.data_preprocessing import load_data

from src.decision_tree import train_decision_tree

from src.random_forest import train_random_forest

from src.visualize import (
    visualize_tree,
    feature_importance_plot,
    confusion_matrix_plot
)


def main():

    print("\nSTARTING W3D3 PIPELINE\n")


    # Setup MLflow
    setup_mlflow()


    # Load data for visualization
    X_train, X_test, y_train, y_test = load_data()



    # Train Decision Tree
    print("\nTraining Decision Tree...")

    decision_tree_model = train_decision_tree()



    # Generate Decision Tree plot
    visualize_tree(
        decision_tree_model,
        X_train.columns
    )



    # Feature importance plot
    feature_importance_plot(
        decision_tree_model,
        X_train.columns
    )



    # Confusion Matrix
    confusion_matrix_plot(
        decision_tree_model,
        X_test,
        y_test
    )



    # Train Random Forest
    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest()



    # Feature importance for Random Forest also
    feature_importance_plot(
        random_forest_model,
        X_train.columns
    )


    print("\nPIPELINE COMPLETED SUCCESSFULLY")



if __name__ == "__main__":
    main()