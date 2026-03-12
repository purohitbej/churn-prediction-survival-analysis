"""
test_api.py
-----------
Basic smoke tests for the Flask API.
Run with: pytest tests/test_api.py -v
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np

from api.app import create_app


# Fixtures
# ========
@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return app.test_client()


SAMPLE_CUSTOMER = {
    "PreferredLoginDevice": "Phone",
    "CityTier": 1,
    "WarehouseToHome": 12.0,
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Male",
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 3,
    "MaritalStatus": "Married",
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 2.0,
    "OrderCount": 3.0,
    "DaySinceLastOrder": 5.0,
    "Tenure": 10.0,
    "CashbackAmount": 180.0,
}


# Health check
# ============
def test_health_endpoint_structure(client):
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    assert "status" in data
    assert "model_loaded" in data


# Single prediction
# =================
@patch("api.app.load_pipeline")
@patch("api.app.load_threshold")
@patch("api.app.customer_to_dataframe")
def test_predict_single_success(mock_df, mock_thresh, mock_pipeline, client):
    import pandas as pd

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_pipeline.return_value = mock_model
    mock_thresh.return_value = 0.5
    mock_df.return_value = pd.DataFrame({"feature": [1]})

    response = client.post(
        "/predict",
        data=json.dumps(SAMPLE_CUSTOMER),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "threshold_used" in data
    assert "risk_level" in data
    assert data["churn_prediction"] == 1
    assert data["risk_level"] == "High"


def test_predict_missing_body(client):
    response = client.post("/predict", data="", content_type="application/json")
    assert response.status_code == 400


# Batch prediction
# ================
@patch("api.app.load_pipeline")
@patch("api.app.load_threshold")
@patch("api.app.customer_to_dataframe")
def test_predict_batch_success(mock_df, mock_thresh, mock_pipeline, client):
    import pandas as pd

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    mock_pipeline.return_value = mock_model
    mock_thresh.return_value = 0.5
    mock_df.return_value = pd.DataFrame({"feature": [1]})

    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    response = client.post(
        "/predict/batch",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    assert data["total_customers"] == 2
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert "churn_rate" in data


def test_predict_batch_empty_list(client):
    payload = {"customers": []}
    response = client.post(
        "/predict/batch",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_predict_batch_missing_key(client):
    response = client.post(
        "/predict/batch",
        data=json.dumps({"wrong_key": []}),
        content_type="application/json",
    )
    assert response.status_code == 400


# Risk level helper
# =================
def test_risk_levels():
    from api.utils import risk_level

    assert risk_level(0.1) == "Low"
    assert risk_level(0.4) == "Medium"
    assert risk_level(0.6) == "High"
    assert risk_level(0.9) == "Critical"
