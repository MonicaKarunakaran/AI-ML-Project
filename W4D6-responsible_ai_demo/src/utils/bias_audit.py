import pandas as pd


def detect_bias(df: pd.DataFrame, protected_cols: list) -> pd.DataFrame:
    """
    Performs a basic audit of six common bias types.

    Bias types:
    1. Sampling bias
    2. Measurement bias
    3. Representation bias
    4. Historical bias
    5. Label bias
    6. Aggregation bias

    Note:
    Some bias types cannot be proven from a dataset alone.
    This function provides audit indicators that require
    human/domain review.
    """

    report = []

    n_total = len(df)

    for col in protected_cols:

        if col not in df.columns:
            report.append({
                "protected_attribute": col,
                "bias_type": "Data availability",
                "finding": "Column not found",
                "severity": "High"
            })
            continue

        counts = df[col].value_counts(dropna=False)

        # Representation / sampling audit
        for value, count in counts.items():

            percentage = (count / n_total) * 100

            if percentage < 5:
                severity = "High"
            elif percentage < 10:
                severity = "Medium"
            else:
                severity = "Low"

            report.append({
                "protected_attribute": col,
                "bias_type": "Representation bias",
                "group": value,
                "count": int(count),
                "percentage": round(percentage, 2),
                "finding": (
                    f"Group represents {percentage:.2f}% of the dataset"
                ),
                "severity": severity
            })

        # Missing-value / measurement indicator
        missing_count = df[col].isna().sum()

        report.append({
            "protected_attribute": col,
            "bias_type": "Measurement bias",
            "group": "All",
            "count": int(missing_count),
            "percentage": round(
                (missing_count / n_total) * 100, 2
            ),
            "finding": (
                f"{missing_count} missing values detected"
            ),
            "severity": "Medium" if missing_count > 0 else "Low"
        })

        # Sampling bias requires external population information.
        report.append({
            "protected_attribute": col,
            "bias_type": "Sampling bias",
            "group": "All",
            "count": n_total,
            "percentage": 100.0,
            "finding": (
                "Requires comparison with the target real-world "
                "population distribution"
            ),
            "severity": "Review"
        })

        # Historical bias requires analysis of historical labels.
        report.append({
            "protected_attribute": col,
            "bias_type": "Historical bias",
            "group": "All",
            "count": n_total,
            "percentage": 100.0,
            "finding": (
                "Requires checking whether historical outcomes "
                "contain discriminatory patterns"
            ),
            "severity": "Review"
        })

        # Label bias requires checking label quality.
        report.append({
            "protected_attribute": col,
            "bias_type": "Label bias",
            "group": "All",
            "count": n_total,
            "percentage": 100.0,
            "finding": (
                "Requires checking whether target labels were "
                "consistently assigned across groups"
            ),
            "severity": "Review"
        })

        # Aggregation bias
        report.append({
            "protected_attribute": col,
            "bias_type": "Aggregation bias",
            "group": "All",
            "count": n_total,
            "percentage": 100.0,
            "finding": (
                "Check whether one model is being applied to "
                "heterogeneous groups with different patterns"
            ),
            "severity": "Review"
        })

    return pd.DataFrame(report)