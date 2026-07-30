# Week 3 Day 3: Decision Trees & Random Forest Classifier Pipeline

An end-to-end Machine Learning pipeline utilizing **Decision Tree** and **Random Forest** algorithms for breast cancer classification. This project incorporates data preprocessing, model evaluation, visualization generation, unit testing, and experiment tracking using **MLflow**.

---

## 📂 Project Structure

```text
W3D3-DecisionTrees-RandomForest/
├── models/                       # Saved trained model artifacts (.pkl)
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── plots/                        # Generated visualization artifacts
│   ├── decision_tree.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration and path management
│   ├── data_preprocessing.py     # Data loading and preprocessing pipeline
│   ├── decision_tree.py          # Decision Tree model training script
│   ├── evaluate.py               # Evaluation metrics utilities
│   ├── mlflow_logger.py          # MLflow logging setup and utility functions
│   ├── random_forest.py          # Random Forest model training script
│   └── visualize.py             # Plotting utilities for decision trees, feature importance, etc.
├── tests/                        # Automated unit tests
│   └── test_models.py
├── .gitignore                    # Version control ignore rules
├── main.py                       # Main pipeline orchestration script
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation