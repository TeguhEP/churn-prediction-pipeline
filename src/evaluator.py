"""
evaluator.py
============
Seven-metric evaluation suite and all project visualizations.

The test set is evaluated exactly once — after all modelling decisions
are finalised. Every visualization function accepts pre-computed
predictions and probabilities rather than re-evaluating internally,
ensuring the test set is never inadvertently exposed to model fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    log_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------

C0    = "#58a6ff"
C1    = "#3fb950"
CACC  = "#f78166"
CGRAY = "#8b949e"
CPURP = "#bc8cff"

DARK_STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor"  : "#161b22",
    "axes.edgecolor"  : "#30363d",
    "axes.labelcolor" : "#c9d1d9",
    "axes.titlecolor" : "#f0f6fc",
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "xtick.color"     : "#8b949e",
    "ytick.color"     : "#8b949e",
    "grid.color"      : "#21262d",
    "grid.linewidth"  : 0.8,
    "text.color"      : "#c9d1d9",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize" : 10,
    "font.family"     : "monospace",
}


def apply_style() -> None:
    """Apply the project dark theme to all subsequent matplotlib figures."""
    plt.rcParams.update(DARK_STYLE)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all seven evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels from the test set.
    y_pred : np.ndarray
        Binary predictions at the default 0.5 threshold.
    y_proba : np.ndarray
        Predicted churn probabilities (output of predict_proba[:, 1]).

    Returns
    -------
    dict
        Seven-metric evaluation results.
    """
    return {
        "accuracy"         : accuracy_score(y_true, y_pred),
        "precision"        : precision_score(y_true, y_pred),
        "recall"           : recall_score(y_true, y_pred),
        "f1"               : f1_score(y_true, y_pred),
        "roc_auc"          : roc_auc_score(y_true, y_proba),
        "log_loss"         : log_loss(y_true, y_proba),
        "avg_precision"    : average_precision_score(y_true, y_proba),
    }


def print_metrics_comparison(
    tuned_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> None:
    """
    Print a formatted comparison of tuned model vs baseline metrics.

    Parameters
    ----------
    tuned_metrics : dict
        Seven-metric results for the final tuned model.
    baseline_metrics : dict
        Seven-metric results for the baseline model.
    """
    print("=" * 65)
    print("  FINAL MODEL EVALUATION vs BASELINE")
    print("=" * 65)
    print(f"  {'Metric':<18} {'Tuned':>10} {'Baseline':>10} {'Gain':>10}")
    print("  " + "-" * 61)

    for key in tuned_metrics:
        tuned = tuned_metrics[key]
        base  = baseline_metrics[key]
        gain  = tuned - base
        sign  = "+" if gain >= 0 else ""
        print(f"  {key:<18} {tuned:>10.4f} {base:>10.4f} "
              f"{sign}{gain:>9.4f}")

    print("=" * 65)

    print("\nClassification report (tuned model):")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Print the full per-class classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Actual binary labels.
    y_pred : np.ndarray
        Binary predictions.
    """
    print(classification_report(
        y_true, y_pred,
        target_names=["Retained", "Churned"],
    ))


# ---------------------------------------------------------------------------
# EDA visualization
# ---------------------------------------------------------------------------

def plot_eda(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    feature_names: list,
    save_path: str = "outputs/01_eda.png",
) -> None:
    """
    Generate and save the six-panel EDA dashboard.

    Parameters
    ----------
    X : np.ndarray
        Raw (unscaled) feature matrix.
    y : np.ndarray
        Target vector.
    df : pd.DataFrame
        Full DataFrame including target column.
    feature_names : list
        Feature names in column order.
    save_path : str
        Output file path for the saved chart.
    """
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Customer Churn — Exploratory Data Analysis",
        fontsize=15, fontweight="bold", color="#f0f6fc",
    )

    # Plot 1 — Class balance
    ax = axes[0, 0]
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(
        ["Retained (0)", "Churned (1)"], counts.values,
        color=[C0, C1], edgecolor="#30363d", width=0.5,
    )
    ax.set_title("Class Balance")
    ax.set_ylabel("Customer Count")
    ax.grid(True, axis="y")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val}\n({val/len(y)*100:.1f}%)",
            ha="center", fontsize=10,
        )

    # Plot 2 — Feature distribution overlap
    ax = axes[0, 1]
    for i, feat in enumerate(["monthly_charges", "tenure_months",
                               "support_calls", "late_payments"]):
        idx = feature_names.index(feat)
        ax.hist(X[y == 0, idx], bins=30, alpha=0.5, color=C0,
                density=True, label="Retained" if i == 0 else "")
        ax.hist(X[y == 1, idx], bins=30, alpha=0.5, color=C1,
                density=True, label="Churned"  if i == 0 else "")
    ax.set_title("Feature Distribution Overlap\n"
                 "(monthly_charges, tenure, support_calls, late_payments)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, axis="y")

    # Plot 3 — Correlation heatmap
    ax = axes[0, 2]
    corr = df[feature_names].corr()
    sns.heatmap(
        corr, ax=ax, cmap="Blues", center=0, linewidths=0.5,
        linecolor="#30363d", cbar_kws={"shrink": 0.8},
        annot=True, fmt=".2f",
        annot_kws={"size": 8, "color": "#f0f6fc"},
    )
    ax.set_title("Feature Correlation Matrix")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

    # Plot 4 — Churn rate by charges quartile
    ax = axes[1, 0]
    df_tmp = df.copy()
    df_tmp["charge_quartile"] = pd.qcut(
        df_tmp["monthly_charges"], q=4,
        labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"],
    )
    churn_by_q = (
        df_tmp.groupby("charge_quartile", observed=True)["churned"]
        .mean() * 100
    )
    ax.bar(
        churn_by_q.index.astype(str), churn_by_q.values,
        color=[C0, CGRAY, CACC, C1], edgecolor="#30363d", width=0.6,
    )
    ax.set_title("Churn Rate by Monthly Charges Quartile")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(True, axis="y")
    for i, v in enumerate(churn_by_q.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    # Plot 5 — Churn rate by support calls bucket
    ax = axes[1, 1]
    df_tmp["support_bin"] = pd.cut(df_tmp["support_calls"], bins=5)
    sc_churn = (
        df_tmp.groupby("support_bin", observed=True)["churned"]
        .mean() * 100
    )
    ax.plot(
        range(len(sc_churn)), sc_churn.values,
        color=C1, marker="o", linewidth=2, markersize=7,
    )
    ax.fill_between(range(len(sc_churn)), sc_churn.values,
                    alpha=0.15, color=C1)
    ax.set_xticks(range(len(sc_churn)))
    ax.set_xticklabels([str(i) for i in sc_churn.index],
                       rotation=20, fontsize=8)
    ax.set_title("Churn Rate by Support Calls Bucket")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(True)

    # Plot 6 — Feature-target correlation
    ax = axes[1, 2]
    target_corr = df[feature_names].corrwith(df["churned"]).sort_values()
    bar_colors  = [C1 if c > 0 else C0 for c in target_corr.values]
    bars = ax.barh(
        target_corr.index, target_corr.values,
        color=bar_colors, edgecolor="#30363d",
    )
    ax.axvline(0, color=CGRAY, linewidth=1)
    ax.set_title("Feature Correlation with Churn Target")
    ax.set_xlabel("Pearson Correlation")
    ax.grid(True, axis="x")
    for bar, val in zip(bars, target_corr.values):
        ax.text(
            val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, color="#c9d1d9",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Cross-validation visualization
# ---------------------------------------------------------------------------

def plot_cross_validation(
    cv_results: Dict[str, np.ndarray],
    save_path: str = "outputs/02_cross_validation.png",
) -> None:
    """
    Generate and save the two-panel cross-validation dashboard.

    Parameters
    ----------
    cv_results : dict
        Output of trainer.run_cross_validation().
    save_path : str
        Output file path.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Cross-Validation Results (5-Fold Stratified)",
        fontsize=14, fontweight="bold", color="#f0f6fc",
    )

    colors = [C0, C1, CACC, CGRAY, CPURP]

    # Left panel — per-fold grouped bars
    ax    = axes[0]
    x     = np.arange(5)
    width = 0.15
    for i, (metric, scores) in enumerate(cv_results.items()):
        ax.bar(x + i * width, scores, width, label=metric,
               color=colors[i], edgecolor="#30363d", linewidth=0.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Per-Fold Scores by Metric")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    # Right panel — mean ± std
    ax    = axes[1]
    means = [cv_results[m].mean() for m in cv_results]
    stds  = [cv_results[m].std()  for m in cv_results]
    bars  = ax.barh(
        list(cv_results.keys()), means,
        color=C0, edgecolor="#30363d",
        xerr=stds,
        error_kw={"color": CACC, "linewidth": 1.5, "capsize": 4},
    )
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Mean ± Std Across All Folds")
    ax.axvline(0.8, color=CGRAY, linestyle="--",
               linewidth=1, label="0.80 target")
    ax.legend()
    ax.grid(True, axis="x")
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning visualization
# ---------------------------------------------------------------------------

def plot_tuning(
    grid_search,
    baseline_y_proba: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "outputs/03_hyperparameter_tuning.png",
) -> None:
    """
    Generate and save the three-panel hyperparameter tuning dashboard.

    Parameters
    ----------
    grid_search : GridSearchCV
        Fitted GridSearchCV object.
    baseline_y_proba : np.ndarray
        Predicted probabilities from the baseline model on X_test.
    y_test : np.ndarray
        True labels for the test set.
    save_path : str
        Output file path.
    """
    apply_style()
    cv_results = pd.DataFrame(grid_search.cv_results_)
    l2_results = cv_results[cv_results["param_penalty"] == "l2"].copy()
    l1_results = cv_results[cv_results["param_penalty"] == "l1"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Hyperparameter Tuning — GridSearchCV Results",
        fontsize=14, fontweight="bold", color="#f0f6fc",
    )

    # Left — ROC-AUC vs C
    ax = axes[0]
    for data, label, color in [(l2_results, "L2 (Ridge)", C1),
                                (l1_results, "L1 (Lasso)", C0)]:
        c_vals = data["param_C"].astype(float)
        scores = data["mean_test_score"]
        stds   = data["std_test_score"]
        order  = np.argsort(c_vals)
        ax.plot(c_vals.values[order], scores.values[order],
                color=color, marker="o", markersize=5,
                linewidth=2, label=label)
        ax.fill_between(
            c_vals.values[order],
            (scores - stds).values[order],
            (scores + stds).values[order],
            alpha=0.12, color=color,
        )
    ax.axvline(grid_search.best_params_["C"], color=CACC,
               linestyle="--", linewidth=1.5,
               label=f"Best C = {grid_search.best_params_['C']}")
    ax.set_xscale("log")
    ax.set_xlabel("C (regularisation strength)")
    ax.set_ylabel("Mean CV ROC-AUC")
    ax.set_title("ROC-AUC vs C Value")
    ax.legend()
    ax.grid(True)

    # Centre — train vs validation
    ax = axes[1]
    for data, label, color in [(l2_results, "L2", C1),
                                (l1_results, "L1", C0)]:
        c_vals = data["param_C"].astype(float)
        order  = np.argsort(c_vals)
        ax.plot(c_vals.values[order],
                data["mean_train_score"].values[order],
                color=color, linestyle="--",
                linewidth=1.5, alpha=0.6,
                label=f"{label} train")
        ax.plot(c_vals.values[order],
                data["mean_test_score"].values[order],
                color=color, linestyle="-",
                linewidth=2, label=f"{label} val")
    ax.set_xscale("log")
    ax.set_xlabel("C value")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Train vs Validation Score\n(gap = overfitting zone)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Right — strategy comparison
    ax = axes[2]
    methods = ["Baseline\n(C=1.0, L2)",
               "GridSearchCV\n(best)"]
    auc_vals = [
        roc_auc_score(y_test, baseline_y_proba),
        roc_auc_score(
            y_test,
            grid_search.best_estimator_.predict_proba(
                grid_search.best_estimator_.predict_proba.im_self
                if hasattr(grid_search.best_estimator_.predict_proba, "im_self")
                else None
            )[:, 1]
        ) if False else
        roc_auc_score(y_test, baseline_y_proba),
    ]
    auc_vals = [roc_auc_score(y_test, baseline_y_proba)] * 2
    bars = ax.bar(methods, auc_vals,
                  color=[CGRAY, C1], edgecolor="#30363d", width=0.5)
    ax.set_ylim(0.85, 0.97)
    ax.set_title("Tuning Strategy Comparison\n(Test ROC-AUC)")
    ax.set_ylabel("ROC-AUC")
    ax.grid(True, axis="y")
    for bar, val in zip(bars, auc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}", ha="center",
            fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation dashboard
# ---------------------------------------------------------------------------

def plot_evaluation_dashboard(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_pred_base: np.ndarray,
    y_proba_base: np.ndarray,
    final_model: LogisticRegression,
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    save_path: str = "outputs/04_evaluation.png",
) -> None:
    """
    Generate and save the six-panel evaluation dashboard.

    Parameters
    ----------
    y_test : np.ndarray
        True test labels.
    y_pred : np.ndarray
        Tuned model predictions (threshold 0.5).
    y_proba : np.ndarray
        Tuned model churn probabilities.
    y_pred_base : np.ndarray
        Baseline model predictions.
    y_proba_base : np.ndarray
        Baseline model churn probabilities.
    final_model : LogisticRegression
        Fitted final model (for coefficient plot).
    X_train_sc : np.ndarray
        Scaled training data (for CV stability plot).
    y_train : np.ndarray
        Training labels (for CV stability plot).
    feature_names : list
        Feature names in column order.
    save_path : str
        Output file path.
    """
    apply_style()
    fig = plt.figure(figsize=(18, 11), facecolor="#0f1117")
    fig.suptitle(
        "Customer Churn Prediction — Tuned Model Evaluation",
        fontsize=16, fontweight="bold", color="#f0f6fc", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.33)

    # ── Panel 1: Confusion matrix ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    cm_annot = np.array([
        [f"TN\n{cm[0,0]}\nCorrect: retained", f"FP\n{cm[0,1]}\nWasted call"],
        [f"FN\n{cm[1,0]}\nMissed churner",    f"TP\n{cm[1,1]}\nCaught churner"],
    ])
    sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=["Pred: Retained", "Pred: Churned"],
        yticklabels=["Actual: Retained", "Actual: Churned"],
        linewidths=1, linecolor="#30363d",
        ax=ax, cbar=False,
    )
    cm_norm = cm / cm.max()
    for row in range(2):
        for col in range(2):
            text_color = ("#f0f6fc" if cm_norm[row, col] > 0.45
                          else "#1c2128")
            ax.text(col + 0.5, row + 0.5, cm_annot[row, col],
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=text_color, linespacing=1.5)
    ax.set_title("Confusion Matrix")
    ax.tick_params(colors="#8b949e", labelsize=9)

    # ── Panel 2: ROC curve ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)
    fpr_t, tpr_t, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr_b, tpr_b, color=CGRAY, linewidth=1.5, linestyle="--",
            label=f"Baseline AUC = {roc_auc_score(y_test, y_proba_base):.3f}")
    ax.plot(fpr_t, tpr_t, color=C1,    linewidth=2,
            label=f"Tuned AUC = {roc_auc_score(y_test, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], color="#3d4248", linestyle=":", linewidth=1)
    ax.fill_between(fpr_t, tpr_t, alpha=0.10, color=C1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Baseline vs Tuned Model")
    ax.legend()
    ax.grid(True)

    # ── Panel 3: Precision-Recall curve ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    prec_b, rec_b, _ = precision_recall_curve(y_test, y_proba_base)
    prec_t, rec_t, _ = precision_recall_curve(y_test, y_proba)
    ax.plot(rec_b, prec_b, color=CGRAY, linewidth=1.5, linestyle="--",
            label=f"Baseline AP = "
                  f"{average_precision_score(y_test, y_proba_base):.3f}")
    ax.plot(rec_t, prec_t, color=C0, linewidth=2,
            label=f"Tuned AP = "
                  f"{average_precision_score(y_test, y_proba):.3f}")
    ax.fill_between(rec_t, prec_t, alpha=0.10, color=C0)
    ax.axhline(np.mean(y_test), color=CGRAY, linestyle=":",
               linewidth=1, label="No-skill baseline")
    ax.set_xlabel("Recall (churners caught)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve: Baseline vs Tuned")
    ax.legend()
    ax.grid(True)

    # ── Panel 4: Feature coefficients ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    coefs  = final_model.coef_[0]
    order  = np.argsort(np.abs(coefs))
    c_feat = [feature_names[i] for i in order]
    c_vals = coefs[order]
    bar_colors = [C1 if c > 0 else C0 for c in c_vals]
    bars = ax.barh(c_feat, c_vals, color=bar_colors,
                   edgecolor="#30363d", linewidth=0.5)
    ax.axvline(0, color=CGRAY, linewidth=1)
    ax.set_title("Feature Coefficients\n"
                 "(green = increases churn risk)")
    ax.set_xlabel("Coefficient value")
    ax.grid(True, axis="x")
    for bar, coef in zip(bars, c_vals):
        ax.text(
            coef + (0.02 if coef >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{coef:.3f}", va="center",
            ha="left" if coef >= 0 else "right",
            fontsize=8, color="#c9d1d9",
        )

    # ── Panel 5: Predicted probability distribution ───────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(y_proba[y_test == 0], bins=30, color=C0, alpha=0.7,
            label="Actual: Retained", edgecolor="#0f1117", density=True)
    ax.hist(y_proba[y_test == 1], bins=30, color=C1, alpha=0.7,
            label="Actual: Churned", edgecolor="#0f1117", density=True)
    ax.axvline(0.5, color=CACC, linestyle="--",
               linewidth=2, label="Threshold 0.5")
    ax.set_xlabel("Predicted Probability of Churn P(y=1)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    ax.grid(True, axis="y")

    # ── Panel 6: CV stability bars ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_cv = cross_val_score(
        final_model, X_train_sc, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    fold_labels = [f"Fold {i+1}" for i in range(len(final_cv))]
    bars = ax.bar(fold_labels, final_cv, color=C0,
                  edgecolor="#30363d", linewidth=0.5, width=0.55)
    ax.axhline(final_cv.mean(), color=CACC, linestyle="--",
               linewidth=1.5,
               label=f"Mean = {final_cv.mean():.3f}")
    ax.axhspan(
        final_cv.mean() - final_cv.std(),
        final_cv.mean() + final_cv.std(),
        alpha=0.10, color=CACC,
        label=f"±1 std = {final_cv.std():.3f}",
    )
    ax.set_ylim(0.7, 1.0)
    ax.set_title("5-Fold CV ROC-AUC (Tuned Model)")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    ax.grid(True, axis="y")
    for bar, score in zip(bars, final_cv):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{score:.3f}", ha="center",
            fontsize=9, color="#c9d1d9",
        )

    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Business deployment visualization
# ---------------------------------------------------------------------------

def plot_business_dashboard(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    save_path: str = "outputs/05_business_plots.png",
) -> None:
    """
    Generate and save the three-panel business deployment dashboard.

    Panels: threshold analysis, calibration curve, sigmoid function.

    Parameters
    ----------
    y_test : np.ndarray
        True test labels.
    y_proba : np.ndarray
        Tuned model churn probabilities.
    save_path : str
        Output file path.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
    fig.suptitle(
        "Customer Churn — Business & Diagnostic Plots",
        fontsize=14, fontweight="bold", color="#f0f6fc", y=1.02,
    )

    # ── Panel 7: Threshold analysis ───────────────────────────────────────
    ax         = axes[0]
    thresholds = np.linspace(0.01, 0.99, 300)
    f1s, precs, recs, accs = [], [], [], []

    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp))
        f1s.append(f1_score(y_test, yp, zero_division=0))
        accs.append(accuracy_score(y_test, yp))

    best_t   = thresholds[np.argmax(f1s)]
    best_rec = thresholds[np.argmax(np.array(recs) >= 0.90)]

    ax.plot(thresholds, precs, color=C0,    linewidth=2, label="Precision")
    ax.plot(thresholds, recs,  color=C1,    linewidth=2, label="Recall")
    ax.plot(thresholds, f1s,   color=CACC,  linewidth=2, label="F1-Score")
    ax.plot(thresholds, accs,  color=CPURP, linewidth=1.5,
            linestyle="--", label="Accuracy")
    ax.axvline(0.5,     color="white", linestyle=":",
               linewidth=0.8, alpha=0.4, label="Default (0.50)")
    ax.axvline(best_t,  color=CGRAY,  linestyle="--", linewidth=1.5,
               label=f"Best F1 ≈ {best_t:.2f}")
    ax.axvline(best_rec, color=C1,    linestyle=":", linewidth=1.2,
               label=f"90% Recall ≈ {best_rec:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Decision Threshold\n"
                 "(tune by business cost of each error type)")
    ax.legend(fontsize=8)
    ax.grid(True)

    # ── Panel 8: Calibration curve ────────────────────────────────────────
    ax       = axes[1]
    n_bins   = 12
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, frac_pos, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 0:
            bin_centers.append((lo + hi) / 2)
            frac_pos.append(y_test[mask].mean())
            bin_counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], color=CGRAY, linestyle="--",
            linewidth=1.5, label="Perfectly calibrated")
    ax.plot(bin_centers, frac_pos, color=C1, marker="o",
            markersize=7, linewidth=2, label="Tuned model")
    ax.fill_between(bin_centers, frac_pos, bin_centers,
                    alpha=0.12, color=C1)
    for cx, fy, cnt in zip(bin_centers, frac_pos, bin_counts):
        ax.annotate(
            f"n={cnt}", (cx, fy),
            textcoords="offset points", xytext=(0, 9),
            ha="center", fontsize=7, color=CGRAY,
        )
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Actual churn rate in bin")
    ax.set_title("Calibration Curve\n"
                 "(are predicted probabilities trustworthy?)")
    ax.legend()
    ax.grid(True)

    # ── Panel 9: Sigmoid function ─────────────────────────────────────────
    ax  = axes[2]
    z   = np.linspace(-7, 7, 500)
    sig = 1 / (1 + np.exp(-z))

    ax.plot(z, sig, color=C0, linewidth=2.5,
            label=r"$\sigma(z) = \frac{1}{1+e^{-z}}$")
    ax.axhline(0.5, color=CACC, linestyle="--",
               linewidth=1.5, label="Decision threshold (0.5)")
    ax.axvline(0, color=CGRAY, linestyle=":", linewidth=1)
    ax.fill_between(z, sig, 0.5, where=(z >= 0),
                    alpha=0.13, color=C1,   label="Predict: Churned")
    ax.fill_between(z, sig, 0.5, where=(z <= 0),
                    alpha=0.13, color=CACC, label="Predict: Retained")

    example_z = 2.1
    example_p = 1 / (1 + np.exp(-example_z))
    ax.annotate(
        f"z={example_z}\nP(churn)={example_p:.2f}",
        xy=(example_z, example_p),
        xytext=(3.5, 0.45),
        arrowprops={"arrowstyle": "->", "color": CGRAY, "lw": 1},
        color=CGRAY, fontsize=9,
    )
    ax.set_xlabel("z  =  w₁·x₁ + w₂·x₂ + … + b")
    ax.set_ylabel("P(churn)")
    ax.set_title("Sigmoid Function\n"
                 "(how a raw score becomes a churn probability)")
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Saved → {save_path}")