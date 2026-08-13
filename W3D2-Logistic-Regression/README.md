# Logistic Regression Pipeline (Binary & Multiclass)

A modular, production-ready Machine Learning pipeline demonstrating Binary Classification (Breast Cancer dataset) and Multiclass Classification (Iris dataset) using Logistic Regression with MLflow tracking.

## Project Structure

```text
├── data/                  # Output directory for processed datasets
├── outputs/               # Output directory for generated figures and metrics
├── src/                   # Source code directory
│   ├── __init__.py        # Package marker
│   ├── binary.py          # Binary classification training and evaluation
│   ├── config.py          # Centralized path and hyperparameter configuration
│   ├── loader.py          # Data loading and splitting module
│   ├── mlflow_logger.py   # MLflow experiment tracking utilities
│   ├── multiclass.py      # Multiclass classification training and evaluation
│   └── preprocess.py      # Data scaling and feature engineering module
├── .gitignore             # Ignored files and artifacts
├── logistic_regression.py # Main orchestration pipeline script
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies