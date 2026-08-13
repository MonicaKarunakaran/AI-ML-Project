"""
Project configuration settings.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Train-Test split ratio
TEST_SIZE = 0.2

# MLflow Experiment Name
MLFLOW_EXPERIMENT = "W3D3_DecisionTrees_RandomForest"

# Model Paths
DECISION_TREE_MODEL_PATH = "models/decision_tree_model.pkl"
RANDOM_FOREST_MODEL_PATH = "models/random_forest_model.pkl"

# Plot Paths
TREE_PLOT_PATH = "plots/decision_tree.png"
FEATURE_IMPORTANCE_PATH = "plots/feature_importance.png"
CONFUSION_MATRIX_PATH = "plots/confusion_matrix.png"

# Dataset Path
DATASET_PATH = "data/breast_cancer.csv"