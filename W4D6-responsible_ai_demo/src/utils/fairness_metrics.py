import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def _create_dataset(
    df: pd.DataFrame,
    labels,
    protected_attribute: str
):
    """
    Convert a dataframe and labels into an AIF360 BinaryLabelDataset.
    """

    temp = pd.DataFrame({
        "label": labels,
        protected_attribute: df[protected_attribute].astype(int)
    })

    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=temp,
        label_names=["label"],
        protected_attribute_names=[protected_attribute]
    )

    return dataset


def compute_fairness(
    y_true,
    y_pred,
    df,
    protected_attribute="sex"
):
    """
    Compute three fairness metrics using IBM AI Fairness 360.

    Returns:
        dict containing:
        - Disparate Impact
        - Equal Opportunity Difference
        - Statistical Parity Difference
    """

    true_dataset = _create_dataset(
        df,
        y_true,
        protected_attribute
    )

    predicted_dataset = _create_dataset(
        df,
        y_pred,
        protected_attribute
    )

    privileged_groups = [
        {protected_attribute: 1}
    ]

    unprivileged_groups = [
        {protected_attribute: 0}
    ]

    metric = ClassificationMetric(
        true_dataset,
        predicted_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    return {
        "disparate_impact": float(
            metric.disparate_impact()
        ),
        "equal_opportunity_difference": float(
            metric.equal_opportunity_difference()
        ),
        "statistical_parity_difference": float(
            metric.statistical_parity_difference()
        )
    }