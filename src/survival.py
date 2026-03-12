"""
survival.py
-----------
Kaplan-Meier curves, log-rank tests, and Cox Proportional Hazard regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.statistics import proportional_hazard_test

from src.data_preparation import multiple_imputation


# Data preparation for survival analysis
# Different from the classification pipeline, different feature set
# =================================================================

PAYMENT_MODE_GROUPS = {
    "CC": "UPI&CreditCard",
    "Credit Card": "UPI&CreditCard",
    "UPI": "UPI&CreditCard",
    "COD": "COD&EWallet",
    "Cash on Delivery": "COD&EWallet",
    "E wallet": "COD&EWallet",
}

MARITAL_STATUS_GROUPS = {
    "Divorced": "DivorcedOrMarried",
    "Married": "DivorcedOrMarried",
}

ORDER_CAT_GROUPS = {
    "Grocery": "Grocery&Others",
    "Others": "Grocery&Others",
    "Mobile Phone": "Mobile",
}

SURVIVAL_DROP_COLS = [
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "PreferedOrderCat",
    "OrderCount",
    "WarehouseToHome",
    "NumberOfAddress",
    "CashbackAmount",
    "DaySinceLastOrder",
    "Gender",
    "CityTier",
]


def prepare_survival_data(
    file_path: str,
    is_excel: bool = False,
    sheet_name: str | None = None,
    n_imputations: int = 10,
    imputation_index: int = 5,
) -> pd.DataFrame:
    """
    Prepares data for Cox PH regression.

    Key decisions made during analysis
    -----------------------------------
    - Drops features that violated the proportional hazard assumption
      (PreferedOrderCat, Gender, CityTier — shown by Schoenfeld residuals)
    - NumberOfDeviceRegistered is used as a stratification variable
    - Payment modes grouped by similar survival curves (log-rank tests)
    - Marital status: Divorced ≈ Married → collapsed
    - One-hot encodes remaining categoricals with drop_first=True
    """
    if is_excel:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(file_path)

    df = multiple_imputation(df, "Tenure", n_imputations=n_imputations)[
        imputation_index
    ]
    df.drop(SURVIVAL_DROP_COLS, axis=1, inplace=True, errors="ignore")

    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(
        {"Mobile Phone": "Phone"}
    )
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(PAYMENT_MODE_GROUPS)
    df["MaritalStatus"] = df["MaritalStatus"].replace(MARITAL_STATUS_GROUPS)
    df["NumberOfDeviceRegistered"] = df["NumberOfDeviceRegistered"].clip(upper=6)

    df = pd.get_dummies(
        df,
        columns=["PreferredLoginDevice", "PreferredPaymentMode", "MaritalStatus"],
        drop_first=True,
        dtype=float,
    )
    return df


# Kaplan-Meier functions
# ======================
def plot_kaplan_meier_overall(
    time: pd.Series,
    event: pd.Series,
    title: str = "Kaplan-Meier Survival Curve",
) -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(time, event, label="All Customers")

    plt.figure(figsize=(8, 4))
    kmf.plot()
    plt.title(title)
    plt.xlabel("Tenure (months)")
    plt.ylabel("Probability of Survival")
    plt.tight_layout()
    plt.show()


def plot_km_by_group(
    time: pd.Series,
    event: pd.Series,
    groups: pd.Series,
    group_name: str,
) -> None:
    """Plots Kaplan-Meier curves for each category in `groups`."""
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in groups.unique():
        mask = groups == label
        kmf.fit(time[mask], event_observed=event[mask], label=str(label))
        kmf.plot(ax=ax)

    plt.title(f"Survival by {group_name}")
    plt.xlabel("Tenure")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.show()

    result = multivariate_logrank_test(time, groups, event, alpha=0.95)
    result.print_summary()


# Cox PH model
# ============
def fit_cox_model(
    df: pd.DataFrame,
    strata: list[str] | None = None,
) -> CoxPHFitter:
    """
    Fits the final Cox PH model.

    The best specification stratifies on
    NumberOfDeviceRegistered to handle its time-varying effect.

    Parameters
    ----------
    df     : output of prepare_survival_data()
    strata : list of stratification columns

    Returns
    -------
    Fitted CoxPHFitter instance
    """
    if strata is None:
        strata = ["NumberOfDeviceRegistered"]

    cph = CoxPHFitter()
    cph.fit(df, duration_col="Tenure", event_col="Churn", strata=strata)
    cph.print_summary()
    return cph


def plot_subject_hazard(
    cph: CoxPHFitter,
    subject: pd.DataFrame,
) -> None:
    """
    Plots cumulative hazard and survival probability for a single subject,
    with a vertical line marking their current tenure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Cumulative hazard
    cph.predict_cumulative_hazard(subject).plot(ax=axes[0], color="red")
    axes[0].axvline(x=subject["Tenure"].values[0], color="blue", linestyle="--")
    axes[0].legend(["Hazard", "Current Position"])
    axes[0].set(
        xlabel="Tenure", ylabel="Cumulative Hazard", title="Cumulative Hazard Over Time"
    )

    # Survival probability
    cph.predict_survival_function(subject).plot(ax=axes[1], color="orange")
    axes[1].axvline(x=subject["Tenure"].values[0], color="blue", linestyle="--")
    axes[1].legend(["Survival", "Current Position"])
    axes[1].set(
        xlabel="Tenure",
        ylabel="Survival Probability",
        title="Survival Probability Over Time",
    )

    plt.tight_layout()
    plt.show()


def check_ph_assumption(
    cph: CoxPHFitter,
    df: pd.DataFrame,
) -> None:
    """Runs the proportional hazard test and plots Schoenfeld residuals."""
    results = proportional_hazard_test(cph, df, time_transform="rank")
    results.print_summary()

    residuals = cph.compute_residuals(df, kind="scaled_schoenfeld")
    covariates = residuals.columns.tolist()

    fig, axes = plt.subplots(len(covariates), 1, figsize=(8, len(covariates) * 3))
    if len(covariates) == 1:
        axes = [axes]

    for covariate, ax in zip(covariates, axes):
        ax.scatter(residuals.index, residuals[covariate], alpha=0.4, s=10)
        z = np.polyfit(residuals.index, residuals[covariate], 1)
        p = np.poly1d(z)
        ax.plot(residuals.index, p(residuals.index), color="red")
        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_title(covariate)
        ax.set_xlabel("Time")

    plt.tight_layout()
    plt.show()
