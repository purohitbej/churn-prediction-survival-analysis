"""
utils.py
--------
Model loading, feature preparation, and prediction helpers for the API.
"""

import os
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Paths to artefacts produced by scripts/train.py
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "churn_pipeline.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.txt")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.txt")


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Model not found at {PIPELINE_PATH}. "
            "Run `python scripts/train.py` first."
        )
    return joblib.load(PIPELINE_PATH)


@lru_cache(maxsize=1)
def load_encoder() -> OrdinalEncoder:
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Encoder not found at {ENCODER_PATH}.")
    return joblib.load(ENCODER_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    if not os.path.exists(THRESHOLD_PATH):
        return 0.5  # safe default
    with open(THRESHOLD_PATH) as f:
        return float(f.read().strip())


@lru_cache(maxsize=1)
def load_feature_names() -> list[str]:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature names not found at {FEATURES_PATH}.")
    with open(FEATURES_PATH) as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def customer_to_dataframe(customer_dict: dict) -> pd.DataFrame:
    """
    Converts a single customer dict (from the API request) into the
    feature DataFrame that the pipeline expects.

    Mirrors the preprocessing in src/data_preparation.py:
      - Tenure grouping
      - Label consolidation
      - Tenure_Cashback interaction
      - Drop Tenure & CashbackAmount
      - Ordinal encoding
    """
    from src.data_preparation import (
        LOGIN_DEVICE_MAP,
        PAYMENT_MODE_MAP,
        ORDER_CAT_MAP,
        CATEGORICAL_COLS,
    )

    df = pd.DataFrame([customer_dict])

    # Tenure grouping (same as training)
    if "Tenure" in df.columns and df["Tenure"].notna().all():
        df["Tenure"] = df["Tenure"].apply(lambda x: (x // 12) + 1)

    # Label consolidation
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace(LOGIN_DEVICE_MAP)
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(PAYMENT_MODE_MAP)
    df["PreferedOrderCat"] = df["PreferedOrderCat"].replace(ORDER_CAT_MAP)

    # Interaction feature
    tenure = df.get("Tenure", pd.Series([np.nan]))
    cashback = df.get("CashbackAmount", pd.Series([np.nan]))
    df["Tenure_Cashback"] = tenure.values * cashback.values

    # Drop originals
    df.drop(["Tenure", "CashbackAmount"], axis=1, inplace=True, errors="ignore")

    # Ordinal encoding
    encoder = load_encoder()
    df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    # Reorder to match training feature order
    feature_names = load_feature_names()
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan  # imputer will handle it
    df = df[feature_names]

    return df


def risk_level(probability: float) -> str:
    """Maps a churn probability to a human-readable risk tier."""
    if probability < 0.30:
        return "Low"
    elif probability < 0.55:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Critical"
