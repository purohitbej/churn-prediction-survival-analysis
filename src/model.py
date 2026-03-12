"""
model.py
--------
Defines the sklearn Pipeline, evaluation utilities, and the
churn_prediction() diagnostic function used throughout the notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)


# Best hyperparameters found via GridSearchCV (forest.ipynb — final grid)
# =======================================================================
BEST_PARAMS = dict(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight={0: 1, 1: 3.4},
    criterion="entropy",
    max_depth=5,
    max_features="sqrt",
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=20,
    min_samples_split=30,
    min_weight_fraction_leaf=0,
    n_estimators=1000,
    n_jobs=-1,
    oob_score=False,
    random_state=42,
    verbose=0,
    warm_start=False,
)


# Pipeline factory
# ================


def build_pipeline(rf_params: dict | None = None) -> Pipeline:
    """
    Wraps a RandomForestClassifier inside a median-imputer Pipeline.

    Parameters
    ----------
    rf_params : dict of RFC hyperparameters.
                Defaults to the tuned BEST_PARAMS if None.

    Returns
    -------
    sklearn Pipeline  (imputer → RandomForestClassifier)
    """
    params = rf_params if rf_params is not None else BEST_PARAMS
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**params)),
        ]
    )


# Evaluation
# ==========


def churn_prediction(
    algo: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    columns: list[str],
    cf: str = "features",
) -> None:
    """
    Fits the pipeline, prints classification reports for both train and test,
    and renders Confusion Matrix, ROC curves, and Feature Importances.

    Parameters
    ----------
    algo    : an unfitted (or already-fitted) sklearn Pipeline
    cf      : 'features' for tree importances | 'coefficients' for linear coef_
    """
    algo.fit(x_train, y_train)

    predictions = algo.predict(x_test)
    probabilities = algo.predict_proba(x_test)[:, 1]
    train_predictions = algo.predict(x_train)
    train_probabilities = algo.predict_proba(x_train)[:, 1]

    # --- Feature importances ---
    if cf == "coefficients":
        coefficients = pd.DataFrame(algo.named_steps["model"].coef_.ravel())
    else:
        coefficients = pd.DataFrame(algo.named_steps["model"].feature_importances_)

    column_df = pd.DataFrame(columns)
    coef_summary = pd.merge(
        coefficients, column_df, left_index=True, right_index=True, how="left"
    )
    coef_summary.columns = ["coefficients", "features"]
    coef_summary = coef_summary.sort_values("coefficients", ascending=False)

    # --- Console output ---
    print(algo)
    print("\nTest Classification Report:\n", classification_report(y_test, predictions))
    print("Test Accuracy Score:", accuracy_score(y_test, predictions))
    print("-" * 50)
    print(
        "Train Classification Report:\n",
        classification_report(y_train, train_predictions),
    )
    print("Train Accuracy Score:", accuracy_score(y_train, train_predictions))

    test_auc = roc_auc_score(y_test, probabilities)
    train_auc = roc_auc_score(y_train, train_probabilities)
    print("Area under curve (Test): ", test_auc)
    print("Area under curve (Train):", train_auc)

    # --- Plots ---
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    train_fpr, train_tpr, _ = roc_curve(y_train, train_probabilities)

    fig = plt.figure(figsize=(12, 14))

    # Confusion matrices
    ax1 = fig.add_subplot(321)
    sns.heatmap(
        confusion_matrix(y_test, predictions), fmt="d", annot=True, cmap="Blues", ax=ax1
    )
    ax1.set_title("Test Confusion Matrix")
    ax1.set_ylabel("True Values")
    ax1.set_xlabel("Predicted Values")

    ax2 = fig.add_subplot(322)
    sns.heatmap(
        confusion_matrix(y_train, train_predictions),
        fmt="d",
        annot=True,
        cmap="Blues",
        ax=ax2,
    )
    ax2.set_title("Train Confusion Matrix")
    ax2.set_ylabel("True Values")
    ax2.set_xlabel("Predicted Values")

    # ROC curves
    ax3 = fig.add_subplot(323)
    ax3.plot(fpr, tpr, color="orange", lw=1, label=f"AUC: {test_auc:.3f}")
    ax3.plot([0, 1], [0, 1], color="blue", lw=1, linestyle="--")
    ax3.set(
        xlim=[0, 1],
        ylim=[0, 1],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC (Test)",
    )
    ax3.legend(loc="lower right")

    ax4 = fig.add_subplot(324)
    ax4.plot(train_fpr, train_tpr, color="orange", lw=1, label=f"AUC: {train_auc:.3f}")
    ax4.plot([0, 1], [0, 1], color="blue", lw=1, linestyle="--")
    ax4.set(
        xlim=[0, 1],
        ylim=[0, 1],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC (Train)",
    )
    ax4.legend(loc="lower right")

    # Feature importances
    ax5 = fig.add_subplot(313)
    sns.barplot(x=coef_summary["features"], y=coef_summary["coefficients"], ax=ax5)
    ax5.set_title("Feature Importances")
    ax5.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()


def get_optimal_threshold(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Computes the probability threshold that maximises F1 score
    on the provided dataset.

    Returns
    -------
    float : optimal threshold
    """
    from sklearn.metrics import precision_recall_curve

    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = np.where(
        (precision + recall) == 0,
        0,
        2 * (precision * recall) / (precision + recall),
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Optimal threshold: {best_threshold:.4f}  |  F1: {f1_scores[best_idx]:.4f}")
    return float(best_threshold)
