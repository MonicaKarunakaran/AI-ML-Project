# W3D4 - SVM & KNN

## Objective

Compare Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classification models using Scikit-Learn.

## Features

- Breast Cancer Dataset
- StandardScaler
- SVM Classifier
- KNN Classifier
- Model Evaluation
- Confusion Matrix
- MLflow Tracking
- Model Saving
- Unit Testing

## Run

```bash
pip install -r requirements.txt

python src/pipeline.py
```

Start MLflow

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000

Task Structure

aiml-w3-svm-knn/
│
├─ .gitignore              # Version control ignore rules
├─ README.md               # Project documentation
├─ requirements.txt        # Project dependencies
├─ pyproject.toml          # optional – for poetry / pip‑tools
│
├─ data/                   # (git‑ignore) – raw CSV will be downloaded at runtime
│
├─ src/
│   ├─ __init__.py
│   ├─ pipeline.py         # LangGraph orchestration + main() entry‑point
│   ├─ models.py           # SVM & KNN helper classes
│   ├─ utils.py            # CrewAI data loader, scaler, MLflow logger
│   └─ evaluation.py       # Ragas wrapper
│
├─ tests/
│   ├─ __init__.py
│   └─ test_pipeline.py    # simple sanity checks (MLflow run count, accuracy > 0)
│
└─ docs/
    └─ mlflow_runs.png      # screenshot you will attach after the run
```