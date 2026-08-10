# W4D6 — Responsible AI: Bias, Fairness & Transparency

## Overview

This project demonstrates a Responsible AI workflow using the UCI Adult Income dataset.

The project evaluates a Logistic Regression model beyond predictive accuracy by auditing bias, measuring fairness, explaining predictions and applying a bias mitigation technique.

## Objectives

- Identify six common types of bias
- Measure fairness using IBM AI Fairness 360
- Train a Logistic Regression classifier
- Generate SHAP explanations
- Apply Reweighing mitigation
- Compare fairness before and after mitigation
- Generate a model card

## Project Structure

```text
W4D6-responsible_ai_demo/
│
├── src/
│   ├── data/
│   │   └── loan_data.csv
│   ├── notebooks/
│   │   └── 01_responsible_ai_demo.ipynb
│   ├── utils/
│   │   ├── bias_audit.py
│   │   ├── fairness_metrics.py
│   │   ├── shap_explain.py
│   │   └── model_card.py
│   └── requirements.txt
│
├── docs/
├── README.md
└── .gitignore