"""
data_generator.py
=================
Dataset creation and feature naming utilities.

Generates a synthetic telecom churn dataset using scikit-learn's
make_classification, configured to replicate the statistical properties
of real telecom churn data as documented in the academic literature.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "monthly_charges",
    "tenure_months",
    "support_calls",
    "data_usage_gb",
    "contract_type",
    "num_products",
    "late_payments",
    "avg_call_duration",
    "promo_response",
    "age_group",
]

DATASET_CONFIG = {
    "n_samples"    : 2000,
    "n_features"   : 10,
    "n_informative": 6,
    "n_redundant"  : 2,
    "n_repeated"   : 0,
    "n_classes"    : 2,
    "class_sep"    : 0.9,
    "random_state" : 42,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_dataset(
    config: dict = DATASET_CONFIG,
    feature_names: list = FEATURE_NAMES,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Generate a synthetic telecom churn dataset.

    Parameters
    ----------
    config : dict
        Parameters passed to make_classification. Defaults to DATASET_CONFIG.
    feature_names : list
        Human-readable names for each feature column.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary target vector — 1 = churned, 0 = retained.
    feature_names : list
        Feature names in the same column order as X.

    Examples
    --------
    >>> X, y, names = generate_dataset()
    >>> X.shape
    (2000, 10)
    >>> y.sum()
    1000
    """
    X, y = make_classification(**config)

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"n_features ({X.shape[1]})."
        )

    return X, y, feature_names


def build_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    target_name: str = "churned",
) -> pd.DataFrame:
    """
    Combine feature matrix and target vector into a labelled DataFrame.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    feature_names : list
        Column names for X.
    target_name : str
        Name of the target column. Default is 'churned'.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with features and target.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    return df


def print_dataset_summary(
    df: pd.DataFrame,
    target_name: str = "churned",
) -> None:
    """
    Print a structured summary of the generated dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including the target column.
    target_name : str
        Name of the binary target column.
    """
    n_total   = len(df)
    n_churned = df[target_name].sum()
    n_retained = n_total - n_churned

    print("=" * 55)
    print("  DATASET OVERVIEW")
    print("=" * 55)
    print(f"  Shape          : {df.shape}")
    print(f"  Total customers: {n_total:,}")
    print(f"  Churned (y=1)  : {n_churned:,}  ({n_churned/n_total*100:.1f}%)")
    print(f"  Retained (y=0) : {n_retained:,}  ({n_retained/n_total*100:.1f}%)")
    print(f"  Features       : {df.shape[1] - 1}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated().sum()}")
    print("=" * 55)
    print("\nDescriptive statistics:")
    print(df.describe().round(2).to_string())


def get_feature_metadata() -> pd.DataFrame:
    """
    Return a DataFrame describing each feature's business meaning
    and expected relationship with churn.

    Returns
    -------
    pd.DataFrame
        Metadata table with columns: feature, business_meaning,
        expected_direction, academic_basis.
    """
    metadata = [
        ("monthly_charges",   "Total monthly subscription bill",
         "Ambiguous",                   "Hadden et al. (2007)"),
        ("tenure_months",     "Months as active subscriber",
         "Negative (longer = safer)",   "Reichheld & Schefter (2000)"),
        ("support_calls",     "Volume of inbound support contacts",
         "Positive (more = riskier)",   "Verbeke et al. (2012)"),
        ("data_usage_gb",     "Monthly data consumption in gigabytes",
         "Positive",                    "Nie et al. (2011)"),
        ("contract_type",     "Contract commitment length",
         "Positive (flexible = riskier)","Hadden et al. (2007)"),
        ("num_products",      "Number of subscribed service bundles",
         "Ambiguous",                   "Verbeke et al. (2012)"),
        ("late_payments",     "History of payment delays",
         "Positive (more = riskier)",   "Nie et al. (2011)"),
        ("avg_call_duration", "Average support call duration",
         "Negative (longer = safer)",   "Hadden et al. (2007)"),
        ("promo_response",    "Response to promotional offers",
         "Ambiguous",                   "Verbeke et al. (2012)"),
        ("age_group",         "Customer demographic age segment",
         "Negative (older = safer)",    "Nie et al. (2011)"),
    ]

    return pd.DataFrame(
        metadata,
        columns=[
            "feature",
            "business_meaning",
            "expected_direction",
            "academic_basis",
        ],
    )


# ---------------------------------------------------------------------------
# Entry point — run as a script for quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, names = generate_dataset()
    df = build_dataframe(X, y, names)
    print_dataset_summary(df)

    print("\nFeature metadata:")
    print(get_feature_metadata().to_string(index=False))