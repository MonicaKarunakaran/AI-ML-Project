# W4D2 - Bias Variance Tradeoff & Regularisation

## Objective

This project demonstrates the Bias-Variance Tradeoff using Linear Regression and Ridge Regression (L2 Regularisation).

---

## Project Structure

```
W4D2-BiasVariance-Regularisation/
│
├── data/
├── src/
├── tests/
├── plots/
├── mlflow/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- MLflow

---

## Models Used

- Linear Regression
- Ridge Regression

---

## Evaluation Metrics

- MSE
- RMSE
- MAE
- R² Score

---

## How to Run

### Generate Dataset

```bash
python -m src.pipeline
```

### Train Models

```bash
python -m src.train
```

### Evaluate Models

```bash
python -m src.evaluate
```

### Open MLflow

```bash
mlflow ui
```

Visit

```
http://127.0.0.1:5000
```

### Run Tests

```bash
pytest
```

---

## Output

The project generates

- Synthetic Dataset
- Trained Models
- MLflow Runs
- Prediction Plots
- Bias-Variance Comparison Plot

---

## Learning Outcome

- Understanding Bias and Variance
- Comparing Linear Regression and Ridge Regression
- MLflow Experiment Tracking
- Model Evaluation
- Project Modularization