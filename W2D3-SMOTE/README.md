# W2D3: Handling Imbalanced Data using SMOTE

## Objective

The objective of this project is to learn how to handle imbalanced datasets using the SMOTE (Synthetic Minority Over-sampling Technique) algorithm. SMOTE creates synthetic samples for the minority class, helping improve machine learning model performance.

---

## Dataset

**Breast Cancer Wisconsin Dataset**

- Source: Scikit-learn
- Samples: 569
- Features: 30
- Target Classes:
  - 0 = Malignant
  - 1 = Benign

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- VS Code
- Git & GitHub

---

## Project Workflow

1. Load the Breast Cancer dataset.
2. Explore the dataset.
3. Check class distribution.
4. Split the dataset into training and testing sets.
5. Train a Logistic Regression model before SMOTE.
6. Evaluate the model.
7. Apply SMOTE to the training data.
8. Train the model again using balanced data.
9. Compare the performance before and after SMOTE.
10. Visualize the results.

---

## Outputs

The following files are generated in the `outputs` folder:

- before_smote.png
- after_smote.png
- confusion_matrix.png

---

## Machine Learning Model

- Logistic Regression

---

## Project Structure

```
W2D3-SMOTE/
│
├── smote_handling.py
├── README.md
├── requirements.txt
└── outputs/
    ├── before_smote.png
    ├── after_smote.png
    └── confusion_matrix.png
```

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python smote_handling.py
```

---

## Learning Outcomes

- Understood the concept of imbalanced datasets.
- Learned how SMOTE balances minority classes.
- Applied SMOTE correctly on the training dataset.
- Compared model performance before and after SMOTE.
- Visualized the class distribution and confusion matrix.

---