"""
trainer.py
==========
Baseline model, cross-validation, hyperparameter tuning,
and final model training.

All model fitting is performed exclusively on training data.
The test set is never passed to any function in this module —
enforcing the single-use evaluation protocol established in Part 5.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from scipy.stats import loguniform
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "C"           : 1.0,
    "solver"      : "lbfgs",
    "max_iter"    : 1000,
    "random_state": 42,
}

CV_CONFIG = {
    "n_splits"    : 5,
    "shuffle"     : True,
    "random_state": 42,
}

CV_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

GRID_SEARCH_PARAMS = {
    "C"      : [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    "penalty": ["l1", "l2"],
    "solver" : ["liblinear", "saga"],
}

RANDOM_SEARCH_PARAMS = {
    "C"      : loguniform(1e-3, 1e2),
    "penalty": ["l1", "l2"],
    "solver" : ["liblinear", "saga"],
}


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict = BASELINE_CONFIG,
) -> LogisticRegression:
    """
    Train a logistic regression model with default parameters.

    The baseline establishes the performance floor against which every
    subsequent improvement is measured. It uses scikit-learn's default
    parameters with two exceptions: max_iter is raised to 1000 to ensure
    convergence, and random_state is set for reproducibility.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training target vector.
    config : dict
        Model parameters. Defaults to BASELINE_CONFIG.

    Returns
    -------
    LogisticRegression
        Fitted baseline model.
    """
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    model: LogisticRegression,
    X_train: np.ndarray,
    y_train: np.ndarray,
    metrics: list = CV_METRICS,
    cv_config: dict = CV_CONFIG,
) -> Dict[str, np.ndarray]:
    """
    Run stratified K-fold cross-validation across multiple metrics.

    Each metric is evaluated independently across all K folds, producing
    a distribution of scores that characterises both expected performance
    and fold-to-fold variability. All evaluation is performed exclusively
    on training data — the test set is never seen here.

    Parameters
    ----------
    model : LogisticRegression
        Unfitted or fitted model — cross_val_score refits internally.
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training target vector.
    metrics : list
        Scoring metrics to evaluate. Defaults to CV_METRICS.
    cv_config : dict
        StratifiedKFold configuration. Defaults to CV_CONFIG.

    Returns
    -------
    dict
        Keys are metric names, values are np.ndarray of fold scores.
    """
    cv = StratifiedKFold(**cv_config)
    results = {}

    for metric in metrics:
        scores = cross_val_score(
            model, X_train, y_train,
            cv      = cv,
            scoring = metric,
            n_jobs  = -1,
        )
        results[metric] = scores

    return results


def print_cv_summary(cv_results: Dict[str, np.ndarray]) -> None:
    """
    Print a formatted cross-validation results table.

    Parameters
    ----------
    cv_results : dict
        Output of run_cross_validation().
    """
    print("=" * 70)
    print("  CROSS-VALIDATION RESULTS (5-Fold Stratified)")
    print("=" * 70)
    print(f"  {'Metric':<14} {'Fold 1':>7} {'Fold 2':>7} {'Fold 3':>7} "
          f"{'Fold 4':>7} {'Fold 5':>7} {'Mean':>7} {'Std':>7}")
    print("  " + "-" * 66)

    for metric, scores in cv_results.items():
        fold_scores = "  ".join([f"{s:.3f}" for s in scores])
        print(f"  {metric:<14} {fold_scores}  "
              f"{scores.mean():.4f}  {scores.std():.4f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict = GRID_SEARCH_PARAMS,
    cv_config: dict  = CV_CONFIG,
    scoring: str     = "roc_auc",
) -> GridSearchCV:
    """
    Exhaustive hyperparameter search over a discrete parameter grid.

    Evaluates every combination in param_grid using K-fold
    cross-validation on the training set. With 9 C values × 2 penalties
    × 2 solvers = 36 combinations and 5 folds, this performs 180 total
    model fits.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training target vector.
    param_grid : dict
        Parameter values to search. Defaults to GRID_SEARCH_PARAMS.
    cv_config : dict
        StratifiedKFold configuration. Defaults to CV_CONFIG.
    scoring : str
        Metric to optimise. Default 'roc_auc' — most stable metric.

    Returns
    -------
    GridSearchCV
        Fitted search object with best_params_ and best_score_ attributes.
    """
    cv = StratifiedKFold(**cv_config)

    grid_search = GridSearchCV(
        estimator          = LogisticRegression(max_iter=2000,
                                                random_state=42),
        param_grid         = param_grid,
        cv                 = cv,
        scoring            = scoring,
        n_jobs             = -1,
        verbose            = 0,
        return_train_score = True,
    )
    grid_search.fit(X_train, y_train)

    return grid_search


def run_random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: dict = RANDOM_SEARCH_PARAMS,
    n_iter: int               = 60,
    cv_config: dict           = CV_CONFIG,
    scoring: str              = "roc_auc",
) -> RandomizedSearchCV:
    """
    Randomised hyperparameter search over continuous parameter distributions.

    Samples n_iter combinations from param_distributions using a
    log-uniform distribution for C — enabling exploration of the full
    continuous range between 0.001 and 100, not just discrete grid points.
    With 60 iterations and 5 folds, this performs 300 total model fits.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.
    y_train : np.ndarray
        Training target vector.
    param_distributions : dict
        Distributions to sample from. Defaults to RANDOM_SEARCH_PARAMS.
    n_iter : int
        Number of parameter combinations to sample. Default 60.
    cv_config : dict
        StratifiedKFold configuration. Defaults to CV_CONFIG.
    scoring : str
        Metric to optimise. Default 'roc_auc'.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object with best_params_ and best_score_ attributes.
    """
    cv = StratifiedKFold(**cv_config)

    random_search = RandomizedSearchCV(
        estimator           = LogisticRegression(max_iter=2000,
                                                  random_state=42),
        param_distributions = param_distributions,
        n_iter              = n_iter,
        cv                  = cv,
        scoring             = scoring,
        n_jobs              = -1,
        random_state        = 42,
        verbose             = 0,
        return_train_score  = True,
    )
    random_search.fit(X_train, y_train)

    return random_search


def print_tuning_summary(
    baseline_score: float,
    grid_search: GridSearchCV,
    random_search: RandomizedSearchCV,
) -> None:
    """
    Print a structured comparison of all tuning strategies.

    Parameters
    ----------
    baseline_score : float
        Baseline CV ROC-AUC score for comparison.
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    random_search : RandomizedSearchCV
        Fitted RandomizedSearchCV object.
    """
    print("=" * 60)
    print("  HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)
    print(f"  {'Strategy':<22} {'Best config':<22} {'CV ROC-AUC':>10}")
    print("  " + "-" * 56)
    print(f"  {'Baseline':<22} {'C=1.0, L2, lbfgs':<22} "
          f"{baseline_score:>10.4f}")
    print(f"  {'GridSearchCV':<22} "
          f"C={grid_search.best_params_['C']}, "
          f"{grid_search.best_params_['penalty'].upper()}, "
          f"{grid_search.best_params_['solver']:<8} "
          f"{grid_search.best_score_:>10.4f}")
    print(f"  {'RandomizedSearchCV':<22} "
          f"C={random_search.best_params_['C']:.3f}, "
          f"{random_search.best_params_['penalty'].upper()}, "
          f"{random_search.best_params_['solver']:<5} "
          f"{random_search.best_score_:>10.4f}")
    print("=" * 60)
    print(f"\n  Best gain vs baseline: "
          f"+{max(grid_search.best_score_, random_search.best_score_) - baseline_score:.4f} "
          f"ROC-AUC")


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: dict,
) -> LogisticRegression:
    """
    Train the final production model using the best hyperparameters
    on the complete training set.

    Unlike cross-validation fold models (which train on 80% of the
    training set), this model is fitted on all 1,600 training customers,
    producing the most stable coefficient estimates available.

    Parameters
    ----------
    X_train : np.ndarray
        Full scaled training feature matrix (all 1,600 customers).
    y_train : np.ndarray
        Full training target vector.
    best_params : dict
        Best hyperparameters from GridSearchCV.best_params_.

    Returns
    -------
    LogisticRegression
        Fitted final model — ready for evaluation and serialisation.
    """
    final_model = LogisticRegression(
        C            = best_params["C"],
        penalty      = best_params["penalty"],
        solver       = best_params["solver"],
        max_iter     = 2000,
        random_state = 42,
    )
    final_model.fit(X_train, y_train)

    return final_model


def get_coefficient_table(
    model: LogisticRegression,
    feature_names: list,
) -> pd.DataFrame:
    """
    Extract and format the model's learned feature coefficients.

    Parameters
    ----------
    model : LogisticRegression
        Fitted logistic regression model.
    feature_names : list
        Feature names in the same column order as the training data.

    Returns
    -------
    pd.DataFrame
        Coefficient table sorted by absolute magnitude (descending),
        with direction and elimination status columns.
    """
    coefs = model.coef_[0]

    df = pd.DataFrame({
        "feature"    : feature_names,
        "coefficient": coefs.round(4),
        "abs_coef"   : np.abs(coefs).round(4),
        "direction"  : [
            "increases churn risk" if c > 0.001
            else "decreases churn risk" if c < -0.001
            else "eliminated by L1"
            for c in coefs
        ],
        "eliminated" : [abs(c) < 0.001 for c in coefs],
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "feature", "coefficient", "direction", "eliminated"]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data_generator import generate_dataset
    from preprocessor   import split_data, scale_features

    X, y, names       = generate_dataset()
    X_tr, X_te, y_tr, y_te = split_data(X, y)
    X_tr_sc, X_te_sc, _    = scale_features(X_tr, X_te)

    baseline = train_baseline(X_tr_sc, y_tr)
    cv_res   = run_cross_validation(baseline, X_tr_sc, y_tr)
    print_cv_summary(cv_res)

    gs = run_grid_search(X_tr_sc, y_tr)
    rs = run_random_search(X_tr_sc, y_tr)
    print_tuning_summary(cv_res["roc_auc"].mean(), gs, rs)

    final = train_final_model(X_tr_sc, y_tr, gs.best_params_)
    print("\nFeature coefficients:")
    print(get_coefficient_table(final, names).to_string(index=False))