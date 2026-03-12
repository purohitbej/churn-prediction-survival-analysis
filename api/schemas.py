"""
schemas.py
----------
Pydantic models for request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Input features for a single customer churn prediction.
    All fields match the raw column names from the dataset.
    Missing values (NaN) are handled by the pipeline's SimpleImputer.
    """

    PreferredLoginDevice: str = Field(
        ..., example="Phone", description="Phone | Computer"
    )
    CityTier: int = Field(..., example=1, description="1 | 2 | 3")
    WarehouseToHome: Optional[float] = Field(
        None, example=12.0, description="Distance from warehouse to home in km"
    )
    PreferredPaymentMode: str = Field(
        ...,
        example="Debit Card",
        description="Debit Card | Credit Card | COD | UPI | E wallet",
    )
    Gender: str = Field(..., example="Male", description="Male | Female")
    HourSpendOnApp: Optional[float] = Field(
        None, example=3.0, description="Average hours spent on the app per day"
    )
    NumberOfDeviceRegistered: int = Field(
        ..., example=3, description="Number of devices registered to this account"
    )
    PreferedOrderCat: str = Field(
        ...,
        example="Laptop & Accessory",
        description="Laptop & Accessory | Mobile | Fashion | Grocery | Others",
    )
    SatisfactionScore: int = Field(
        ..., example=3, description="Customer satisfaction score (1-5)"
    )
    MaritalStatus: str = Field(
        ..., example="Married", description="Married | Single | Divorced"
    )
    NumberOfAddress: int = Field(
        ..., example=2, description="Number of addresses registered"
    )
    Complain: int = Field(
        ...,
        example=0,
        description="Whether customer filed a complaint last month (0 | 1)",
    )
    OrderAmountHikeFromlastYear: Optional[float] = Field(
        None, example=15.0, description="% increase in order amount from last year"
    )
    CouponUsed: Optional[float] = Field(
        None, example=2.0, description="Number of coupons used last month"
    )
    OrderCount: Optional[float] = Field(
        None, example=3.0, description="Number of orders placed last month"
    )
    DaySinceLastOrder: Optional[float] = Field(
        None, example=5.0, description="Days since last order"
    )
    # Tenure is used to create the Tenure_Cashback feature
    Tenure: Optional[float] = Field(
        None, example=10.0, description="Months of tenure with the platform"
    )
    CashbackAmount: Optional[float] = Field(
        None, example=180.0, description="Average cashback amount received"
    )


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    churn_probability: float = Field(
        ..., description="Predicted probability of churn (0.0 - 1.0)"
    )
    churn_prediction: int = Field(
        ...,
        description="Binary prediction (1 = churn, 0 = retain) at optimal threshold",
    )
    threshold_used: float = Field(
        ..., description="The probability threshold applied for the binary decision"
    )
    risk_level: str = Field(
        ..., description="Human-readable risk tier: Low | Medium | High | Critical"
    )


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_customers: int
    predicted_churners: int
    churn_rate: float
