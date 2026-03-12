"""
data_preparation.py
-------------------
Handles all raw data ingestion, cleaning, encoding and
feature engineering steps that were developed during EDA.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# Categorical mappings  (consolidating duplicate labels found during EDA)
# =======================================================================
LOGIN_DEVICE_MAP = {"Mobile Phone": "Phone"}
PAYMENT_MODE_MAP = {
    "Cash on Delivery": "COD",
    "CC": "Credit Card",
}
ORDER_CAT_MAP = {"Mobile Phone": "Mobile"}

# Columns to ordinally encode
CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


# Multiple Imputation  (used by the survival analysis pipeline)
# =============================================================
def multiple_imputation(
    df: pd.DataFrame,
    duration_col: str,
    n_imputations: int = 10,
    random_state: int = 42,
) -> list[pd.DataFrame]:
    """
    Performs MICE-style iterative imputation on numeric columns.

    Parameters
    ----------
    df              : raw DataFrame (must still contain CustomerID if present)
    duration_col    : the time column (Tenure) — clipped to >=0.01 after imputation
    n_imputations   : number of imputed datasets to return
    random_state    : base random seed; each imputation increments it by 1

    Returns
    -------
    List of imputed DataFrames (length == n_imputations)
    """
    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputed_datasets = []

    for i in range(n_imputations):
        imp = IterativeImputer(
            max_iter=10,
            random_state=random_state + i,
            initial_strategy="median",
        )
        imputed_array = imp.fit_transform(df[numeric_cols])
        imputed_df = df.copy()
        imputed_df[numeric_cols] = imputed_array
        imputed_df[duration_col] = imputed_df[duration_col].clip(lower=0.01)
        imputed_datasets.append(imputed_df)

    return imputed_datasets


# Main data preparation  (used by the classification pipeline)
# ============================================================
def load_and_prepare(
    file_path: str,
    is_excel: bool = False,
    sheet_name: str | None = None,
    encoder: OrdinalEncoder | None = None,
    fit_encoder: bool = True,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """
    Full preprocessing pipeline for the classification model.

    Steps
    -----
    1.  Load CSV or Excel
    2.  Drop CustomerID
    3.  Convert Tenure (months → year-group: (month // 12) + 1)
    4.  Consolidate duplicate category labels
    5.  Engineer Tenure_Cashback interaction feature
    6.  Drop raw Tenure and CashbackAmount (collinear with interaction)
    7.  Ordinally encode categorical columns

    Parameters
    ----------
    file_path    : path to the data file
    is_excel     : True for .xlsx files
    sheet_name   : Excel sheet name (ignored for CSV)
    encoder      : pass a pre-fitted OrdinalEncoder for inference;
                   if None and fit_encoder=True, a new one is created & fitted
    fit_encoder  : whether to fit the encoder on this data (True for training)

    Returns
    -------
    (prepared DataFrame, fitted OrdinalEncoder)
    """
    # 1. Load
    if is_excel:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(file_path)

    # 2. Drop ID
    if "CustomerID" in df.columns:
        df.drop("CustomerID", axis=1, inplace=True)

    # 3. Tenure grouping
    df["Tenure"] = df["Tenure"].apply(lambda x: (x // 12) + 1)

    # 4. Label consolidation
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(LOGIN_DEVICE_MAP)
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(PAYMENT_MODE_MAP)
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(ORDER_CAT_MAP)

    # 5. Interaction feature
    df["Tenure_Cashback"] = df["Tenure"] * df["CashbackAmount"]

    # 6. Drop collinear originals
    df.drop(["Tenure", "CashbackAmount"], axis=1, inplace=True)

    # 7. Ordinal encoding
    if encoder is None:
        encoder = OrdinalEncoder()

    if fit_encoder:
        df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    return df, encoder


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "Churn",
) -> tuple[pd.DataFrame, pd.Series]:
    """Splits a prepared DataFrame into X (features) and y (target)."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y
