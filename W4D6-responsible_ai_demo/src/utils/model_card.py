from pathlib import Path


def create_model_card(
    output_path="model_card.md",
    model_name="Logistic Regression — Adult Income",
    purpose="Adult Income classification",
    training_data="UCI Adult Income dataset",
    performance=None,
    fairness=None,
    explainability="SHAP",
    limitations=None,
    ethical_considerations=None
):
    """
    Generate a Responsible AI model card in Markdown format.
    """

    performance = performance or {}
    fairness = fairness or {}
    limitations = limitations or []
    ethical_considerations = ethical_considerations or []

    performance_text = "\n".join(
        f"- **{key}:** {value:.4f}"
        if isinstance(value, (float, int))
        else f"- **{key}:** {value}"
        for key, value in performance.items()
    )

    fairness_text = "\n".join(
        f"- **{key}:** {value:.4f}"
        if isinstance(value, (float, int))
        else f"- **{key}:** {value}"
        for key, value in fairness.items()
    )

    limitations_text = "\n".join(
        f"- {item}" for item in limitations
    )

    ethics_text = "\n".join(
        f"- {item}" for item in ethical_considerations
    )

    content = f"""# Model Card — {model_name}

## 1. Model Details

**Model:** {model_name}

**Framework:** Scikit-learn

**Version:** 1.0

---

## 2. Intended Use

{purpose}

This model is intended for educational and analytical purposes.
It should not be used as an automated high-stakes decision system.

---

## 3. Training Data

**Dataset:** {training_data}

The model was evaluated using gender as a protected attribute
for Responsible AI fairness analysis.

---

## 4. Model Performance

{performance_text}

---

## 5. Fairness Analysis

{fairness_text}

Fairness metrics were calculated using IBM AI Fairness 360.

---

## 6. Explainability

**Method:** {explainability}

SHAP explanations were generated to understand the contribution
of individual features to model predictions.

---

## 7. Ethical Considerations

{ethics_text}

---

## 8. Limitations

{limitations_text}

---

## 9. Conclusion

The model should be evaluated not only using predictive performance
but also using fairness and explainability measures before
real-world deployment.
"""

    output_path = Path(output_path)

    output_path.write_text(
        content,
        encoding="utf-8"
    )

    print(f"Model card saved to: {output_path}")

    return content