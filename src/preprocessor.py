"""
preprocessor.py
===============
Train / test splitting and feature scaling with leakage prevention.

All preprocessing decisions are documented with their methodological
justification. The key invariant enforced throughout this module:
StandardScaler parameters are fitted exclusively on training data
and applied — never refitted — on test data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_CONFIG = {
    "test_size"   : 0.20,
    "random_state": 42,
    "stratify"    : None,   # set at call time using y
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified train / test split.

    Stratification guarantees that the class proportions of the full
    dataset are preserved in both the training and test sets. Without
    stratification, an unlucky random seed could produce unequal class
    proportions that bias all downstream metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Binary target vector of shape (n_samples,).
    test_size : float
        Proportion of data held out for evaluation. Default 0.20.
    random_state : int
        Seed for reproducibility. Default 42.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split arrays — training and test sets.

    Raises
    ------
    ValueError
        If test_size is not between 0.05 and 0.50.
    """
    if not 0.05 <= test_size <= 0.50:
        raise ValueError(
            f"test_size must be between 0.05 and 0.50, got {test_size}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data and apply to both sets.

    The scaler is fitted exclusively on X_train. The same stored
    parameters (mean and std per feature) are then applied to X_test
    via transform() — never fit_transform(). This prevents data leakage:
    test set distributional properties must not influence the scaling
    parameters used during training.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix — scaler is fitted on this.
    X_test : np.ndarray
        Test feature matrix — scaler is applied but NOT refitted.

    Returns
    -------
    X_train_sc : np.ndarray
        Scaled training features (mean=0, std=1 per feature).
    X_test_sc : np.ndarray
        Scaled test features (using training mean and std).
    scaler : StandardScaler
        Fitted scaler object — must be serialised alongside the model
        for production deployment.
    """
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, scaler


def verify_scaling(
    X_train_sc: np.ndarray,
    feature_names: list,
    n_features: int = 3,
) -> pd.DataFrame:
    """
    Verify that StandardScaler transformed the training set correctly.

    A correctly scaled training set will show means of approximately
    zero and standard deviations of exactly 1.0 for all features.
    Note: the test set will NOT show exactly 0 mean and 1.0 std —
    this is expected and confirms the scaler was not refitted on test data.

    Parameters
    ----------
    X_train_sc : np.ndarray
        Scaled training feature matrix.
    feature_names : list
        Feature names in column order.
    n_features : int
        Number of features to verify. Default 3.

    Returns
    -------
    pd.DataFrame
        Verification table showing post-scaling mean and std per feature.
    """
    n = min(n_features, X_train_sc.shape[1])
    verification = pd.DataFrame({
        "feature"  : feature_names[:n],
        "post_mean": X_train_sc[:, :n].mean(axis=0).round(4),
        "post_std" : X_train_sc[:, :n].std(axis=0).round(4),
        "status"   : [
            "Pass" if abs(m) < 0.01 and abs(s - 1.0) < 0.01
            else "Fail"
            for m, s in zip(
                X_train_sc[:, :n].mean(axis=0),
                X_train_sc[:, :n].std(axis=0),
            )
        ],
    })
    return verification


def print_split_summary(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print a structured summary of the train / test split.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices for training and test sets.
    y_train, y_test : np.ndarray
        Target vectors for training and test sets.
    """
    print("=" * 55)
    print("  TRAIN / TEST SPLIT SUMMARY")
    print("=" * 55)
    print(f"  Training set  : {X_train.shape[0]:>5} customers  "
          f"| churn rate: {y_train.mean()*100:.1f}%")
    print(f"  Test set      : {X_test.shape[0]:>5} customers  "
          f"| churn rate: {y_test.mean()*100:.1f}%")
    print(f"  Total         : {X_train.shape[0]+X_test.shape[0]:>5} customers")
    print(f"  Split ratio   : "
          f"{X_train.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.0f}% / "
          f"{X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])*100:.0f}%")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import generate_dataset, build_dataframe

    X, y, names = generate_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    print_split_summary(X_train, X_test, y_train, y_test)

    print("\nScaling verification (training set):")
    print(verify_scaling(X_train_sc, names).to_string(index=False))