# W3D1 - Linear Regression using Scikit-Learn

## 📌 Objective

Build and evaluate regression models using the California Housing dataset.

The project compares:

- Linear Regression
- Ridge Regression
- Lasso Regression

using Scikit-Learn.

---

## 📂 Project Structure

```
W3D1-Linear-Regression/
│
├── data/
├── plots/
├── results/
├── src/
├── notebooks/
├── tests/
├── linear_regression.py
├── requirements.txt
├── README.md
├── .gitignore
└── cia_reviews.md
```

---

## Dataset

California Housing Dataset

Loaded using:

```python
from sklearn.datasets import fetch_california_housing
```

---

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression

---

## Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## Output

Running the project generates:

### Plots

- Linear Predicted vs Actual
- Linear Residual Plot
- Ridge Predicted vs Actual
- Ridge Residual Plot
- Lasso Predicted vs Actual
- Lasso Residual Plot

### Results

```
results/model_comparison.csv
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python linear_regression.py
```

---

## Author

Monica K