"""
scorer.py
=========
Risk tier assignment, production scoring engine, and output formatting.

Converts continuous model probability scores into the discrete
risk-tiered contact list delivered to the retention team's CRM system.
Includes serialisation utilities for production deployment.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_BOUNDARIES = {
    "high_threshold"  : 0.65,
    "low_threshold"   : 0.35,
}

TIER_LABELS = {
    "high"  : "High Risk",
    "medium": "Medium Risk",
    "low"   : "Low Risk",
}

ACTION_MAP = {
    "High Risk"   : "Personal outreach call",
    "Medium Risk" : "Email campaign",
    "Low Risk"    : "No action",
}


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_risk_tiers(
    y_proba: np.ndarray,
    high_threshold: float = TIER_BOUNDARIES["high_threshold"],
    low_threshold: float  = TIER_BOUNDARIES["low_threshold"],
) -> np.ndarray:
    """
    Convert continuous churn probabilities into discrete risk tier labels.

    Tier boundaries:
    - High Risk   : P(churn) > high_threshold (default 0.65)
    - Medium Risk : low_threshold ≤ P(churn) ≤ high_threshold
    - Low Risk    : P(churn) < low_threshold  (default 0.35)

    The 0.65 upper boundary corresponds approximately to the right edge
    of the central class overlap zone identified in EDA — customers above
    this threshold are in the region of the feature distribution where
    churners clearly dominate. The 0.35 lower boundary is set
    symmetrically to separate the low-risk zone where the calibration
    curve confirms genuine churn rates of 3–11%.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted churn probabilities from the final model.
    high_threshold : float
        Lower bound for High Risk tier. Default 0.65.
    low_threshold : float
        Upper bound for Low Risk tier. Default 0.35.

    Returns
    -------
    np.ndarray of str
        Risk tier label for each customer.
    """
    tiers = np.where(
        y_proba > high_threshold, TIER_LABELS["high"],
        np.where(
            y_proba < low_threshold, TIER_LABELS["low"],
            TIER_LABELS["medium"],
        ),
    )
    return tiers


def assign_recommended_actions(tiers: np.ndarray) -> np.ndarray:
    """
    Map risk tier labels to recommended retention actions.

    Parameters
    ----------
    tiers : np.ndarray
        Risk tier labels from assign_risk_tiers().

    Returns
    -------
    np.ndarray of str
        Recommended action for each customer.
    """
    return np.vectorize(ACTION_MAP.get)(tiers)


# ---------------------------------------------------------------------------
# Output table construction
# ---------------------------------------------------------------------------

def build_scored_output(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    customer_id_prefix: str = "CUST",
    id_start: int           = 5000,
    score_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the complete risk-ranked customer output table.

    This is the production deliverable — the CRM-ready contact list
    sorted from highest to lowest predicted churn probability.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted churn probabilities.
    y_pred : np.ndarray
        Binary predictions at the default 0.5 threshold.
    y_test : np.ndarray
        True labels (included for evaluation — omit in live deployment).
    customer_id_prefix : str
        Prefix for synthetic customer IDs. Default 'CUST'.
    id_start : int
        Starting integer for customer ID generation. Default 5000.
    score_date : str or None
        Date the scores were generated (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame
        Risk-ranked customer output table with 6 production columns.
    """
    if score_date is None:
        score_date = str(date.today())

    tiers   = assign_risk_tiers(y_proba)
    actions = assign_recommended_actions(tiers)

    df = pd.DataFrame({
        "Customer_ID"       : [f"{customer_id_prefix}_{id_start + i}"
                               for i in range(len(y_proba))],
        "Churn_Probability" : y_proba.round(3),
        "Predicted_Label"   : ["Churn" if p == 1 else "Retain"
                               for p in y_pred],
        "Actual_Label"      : ["Churn" if a == 1 else "Retain"
                               for a in y_test],
        "Risk_Tier"         : tiers,
        "Recommended_Action": actions,
        "Score_Date"        : score_date,
    })

    return (df.sort_values("Churn_Probability", ascending=False)
              .reset_index(drop=True))


def print_tier_distribution(scored_df: pd.DataFrame) -> None:
    """
    Print a structured summary of the risk tier distribution.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of build_scored_output().
    """
    total = len(scored_df)
    tier_counts = scored_df["Risk_Tier"].value_counts()

    print("=" * 60)
    print("  RISK TIER DISTRIBUTION")
    print("=" * 60)
    for tier in [TIER_LABELS["high"],
                 TIER_LABELS["medium"],
                 TIER_LABELS["low"]]:
        count = tier_counts.get(tier, 0)
        pct   = count / total * 100
        action = ACTION_MAP[tier]
        print(f"  {tier:<16} : {count:>4} customers "
              f"({pct:>5.1f}%)  →  {action}")

    print("=" * 60)
    print(f"  Total scored   : {total:>4} customers")
    print("=" * 60)


def print_top_customers(
    scored_df: pd.DataFrame,
    n: int = 12,
) -> None:
    """
    Print the top N highest-risk customers.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of build_scored_output(), sorted by Churn_Probability.
    n : int
        Number of top customers to display. Default 12.
    """
    print(f"\nTOP {n} HIGHEST-RISK CUSTOMERS:")
    print(scored_df.head(n)[[
        "Customer_ID",
        "Churn_Probability",
        "Predicted_Label",
        "Actual_Label",
        "Risk_Tier",
        "Recommended_Action",
    ]].to_string(index=False))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: list,
    metrics: dict,
    output_dir: str = "models",
    version: str    = "v1",
) -> None:
    """
    Serialise the fitted model, scaler, feature names, and metadata.

    The model and scaler must always be serialised and deployed as a
    linked pair — they share a version identifier to prevent mismatches
    in production. A mismatched scaler (fitted on a different training
    period than the model) would silently produce incorrect probability
    scores.

    Parameters
    ----------
    model : LogisticRegression
        Fitted final logistic regression model.
    scaler : StandardScaler
        Fitted StandardScaler (training set parameters only).
    feature_names : list
        Feature names in the correct column order for scoring.
    metrics : dict
        Evaluation metrics to store in model metadata.
    output_dir : str
        Directory for serialised files. Default 'models'.
    version : str
        Version tag applied to all artifact filenames. Default 'v1'.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path   = os.path.join(output_dir, f"churn_model_{version}.pkl")
    scaler_path  = os.path.join(output_dir, f"churn_scaler_{version}.pkl")
    names_path   = os.path.join(output_dir, f"feature_names_{version}.json")
    meta_path    = os.path.join(output_dir, f"model_metadata_{version}.json")

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    with open(names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    metadata = {
        "version"              : version,
        "training_date"        : str(date.today()),
        "model_class"          : type(model).__name__,
        "model_params"         : {
            "C"          : model.C,
            "penalty"    : model.penalty,
            "solver"     : model.solver,
            "max_iter"   : model.max_iter,
            "random_state": model.random_state,
        },
        "n_features"           : len(feature_names),
        "feature_names"        : feature_names,
        "features_eliminated"  : [
            name for name, coef
            in zip(feature_names, model.coef_[0])
            if abs(coef) < 0.001
        ],
        "performance_metrics"  : {
            k: round(v, 4) for k, v in metrics.items()
        },
        "tier_boundaries"      : TIER_BOUNDARIES,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 50)
    print("  MODEL ARTIFACTS SAVED")
    print("=" * 50)
    print(f"  Model   → {model_path}")
    print(f"  Scaler  → {scaler_path}")
    print(f"  Names   → {names_path}")
    print(f"  Metadata→ {meta_path}")
    print("=" * 50)


def load_model_artifacts(
    output_dir: str = "models",
    version: str    = "v1",
) -> Tuple[LogisticRegression, StandardScaler, list, dict]:
    """
    Load serialised model, scaler, feature names, and metadata.

    Parameters
    ----------
    output_dir : str
        Directory containing serialised files. Default 'models'.
    version : str
        Version tag of the artifacts to load. Default 'v1'.

    Returns
    -------
    model : LogisticRegression
        Fitted logistic regression model.
    scaler : StandardScaler
        Fitted scaler (training parameters).
    feature_names : list
        Feature names in scoring column order.
    metadata : dict
        Full metadata dictionary including performance metrics.
    """
    model_path  = os.path.join(output_dir, f"churn_model_{version}.pkl")
    scaler_path = os.path.join(output_dir, f"churn_scaler_{version}.pkl")
    names_path  = os.path.join(output_dir, f"feature_names_{version}.json")
    meta_path   = os.path.join(output_dir, f"model_metadata_{version}.json")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(names_path, "r") as f:
        feature_names = json.load(f)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"Loaded model artifacts (version: {version})")
    print(f"  Training date  : {metadata['training_date']}")
    print(f"  ROC-AUC        : {metadata['performance_metrics'].get('roc_auc', 'N/A')}")
    print(f"  Features used  : {metadata['n_features']}")
    print(f"  Features elim. : {metadata['features_eliminated']}")

    return model, scaler, feature_names, metadata


def score_new_customers(
    X_new: np.ndarray,
    model: LogisticRegression,
    scaler: StandardScaler,
    customer_ids: Optional[list] = None,
    score_date: Optional[str]    = None,
) -> pd.DataFrame:
    """
    Score a batch of new, unseen customer records.

    This function replicates the production scoring pipeline:
    apply the stored scaler transformation, evaluate the stored
    model coefficients, assign risk tiers. No refitting occurs.

    Parameters
    ----------
    X_new : np.ndarray
        Raw (unscaled) feature matrix for new customers.
        Must have the same column order as the training data.
    model : LogisticRegression
        Loaded production model from load_model_artifacts().
    scaler : StandardScaler
        Loaded production scaler from load_model_artifacts().
    customer_ids : list or None
        Customer IDs. If None, sequential IDs are generated.
    score_date : str or None
        Scoring date (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame
        Risk-ranked scoring output for the new customer batch.
    """
    if score_date is None:
        score_date = str(date.today())

    if customer_ids is None:
        customer_ids = [f"CUST_NEW_{i}" for i in range(len(X_new))]

    X_scaled = scaler.transform(X_new)
    y_proba  = model.predict_proba(X_scaled)[:, 1]
    y_pred   = model.predict(X_scaled)
    tiers    = assign_risk_tiers(y_proba)
    actions  = assign_recommended_actions(tiers)

    df = pd.DataFrame({
        "Customer_ID"       : customer_ids,
        "Churn_Probability" : y_proba.round(3),
        "Predicted_Label"   : ["Churn" if p == 1 else "Retain"
                               for p in y_pred],
        "Risk_Tier"         : tiers,
        "Recommended_Action": actions,
        "Score_Date"        : score_date,
    })

    return (df.sort_values("Churn_Probability", ascending=False)
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_generator import generate_dataset
    from preprocessor   import split_data, scale_features
    from trainer        import (train_baseline, run_grid_search,
                                train_final_model)
    from evaluator      import compute_metrics

    X, y, names             = generate_dataset()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    X_tr_sc, X_te_sc, scl  = scale_features(X_tr, X_te)

    gs    = run_grid_search(X_tr_sc, y_tr)
    final = train_final_model(X_tr_sc, y_tr, gs.best_params_)

    y_pred  = final.predict(X_te_sc)
    y_proba = final.predict_proba(X_te_sc)[:, 1]
    metrics = compute_metrics(y_te, y_pred, y_proba)

    scored_df = build_scored_output(y_proba, y_pred, y_te)

    print_tier_distribution(scored_df)
    print_top_customers(scored_df)

    save_model_artifacts(final, scl, names, metrics)
