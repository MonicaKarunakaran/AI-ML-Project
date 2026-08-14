# Model Card — Logistic Regression — Adult Income

## 1. Model Details

**Model:** Logistic Regression — Adult Income

**Framework:** Scikit-learn

**Version:** 1.0

---

## 2. Intended Use

Classify whether an individual's annual income is above $50K using demographic and employment-related features.

This model is intended for educational and analytical purposes.
It should not be used as an automated high-stakes decision system.

---

## 3. Training Data

**Dataset:** UCI Adult Income dataset

The model was evaluated using gender as a protected attribute
for Responsible AI fairness analysis.

---

## 4. Model Performance

- **accuracy:** 0.8400
- **precision:** 0.7370
- **recall:** 0.5559
- **f1:** 0.6338
- **roc_auc:** 0.8963

---

## 5. Fairness Analysis

- **disparate_impact:** 0.6368
- **equal_opportunity_difference:** 0.1431
- **statistical_parity_difference:** -0.0774

Fairness metrics were calculated using IBM AI Fairness 360.

---

## 6. Explainability

**Method:** SHAP

SHAP explanations were generated to understand the contribution
of individual features to model predictions.

---

## 7. Ethical Considerations

- Gender was treated as a protected attribute.
- Fairness was evaluated before and after Reweighing.
- SHAP was used to improve model transparency.

---

## 8. Limitations

- The Adult dataset may contain historical and representation bias.
- Fairness metrics depend on the selected protected attribute.
- The model should not be used for high-stakes automated decisions.
- Removing gender from model features does not completely remove proxy bias.

---

## 9. Conclusion

The model should be evaluated not only using predictive performance
but also using fairness and explainability measures before
real-world deployment.
