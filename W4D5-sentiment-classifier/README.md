# Sentiment Classifier

## W4D5 1M Capstone

A binary sentiment classification system that compares Logistic Regression
and Random Forest using TF-IDF text features.

The project also demonstrates MLflow experiment tracking, model serialization,
and a LangGraph-based prediction workflow.

---

## Objective

Build a machine learning pipeline that can:

- Load and clean sentiment data
- Convert text into TF-IDF features
- Train Logistic Regression
- Train Random Forest
- Compare classification performance
- Generate a classification report
- Generate a confusion matrix
- Calculate ROC-AUC
- Track experiments using MLflow
- Save the trained model
- Load the saved model for prediction
- Use LangGraph for the prediction workflow

---

## Project Structure

```text
sentiment-classifier/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── mlflow_utils.py
│   ├── main.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   └── random_forest.py
│   │
│   └── workflow/
│       ├── __init__.py
│       └── sentiment_graph.py
│
├── notebooks/
│   └── 01_explore_and_train.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
└── docs/
    └── RAGAS_report.md