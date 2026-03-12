# Business Problem

## Context

An e-commerce platform is experiencing customer churn — the phenomenon where active customers stop purchasing and disengage from the platform. Retaining an existing customer is significantly cheaper than acquiring a new one (industry estimates range from 5× to 25× depending on the sector), yet churn often goes undetected until it is too late to act.

The platform collects behavioural and transactional data on each customer monthly. The core business question is:

> **Can we identify customers who are likely to churn next month, before they actually leave, so that the retention team can intervene proactively?**

---

## Dataset

The dataset contains **5,630 customer records** with **20 features** covering tenure, engagement behaviour, purchase patterns, logistics, and demographics. The target variable is binary: `Churn = 1` (churned) or `Churn = 0` (retained).

| Feature                       | Description                                                   |
|-------------------------------|---------------------------------------------------------------|
| `Tenure`                      | Months the customer has been active on the platform           |
| `PreferredLoginDevice`        | Device used to log in (Phone / Computer)                      |
| `CityTier`                    | City classification by infrastructure tier (1 / 2 / 3)        |
| `WarehouseToHome`             | Distance from fulfilment warehouse to customer's home (km)    |
| `PreferredPaymentMode`        | Preferred method of payment                                   |
| `Gender`                      | Male / Female                                                 |
| `HourSpendOnApp`              | Average daily hours spent on the app                          |
| `NumberOfDeviceRegistered`    | Devices linked to the account                                 |
| `PreferedOrderCat`            | Most frequently ordered product category                      |
| `SatisfactionScore`           | Self-reported satisfaction (1–5)                              |
| `MaritalStatus`               | Married / Single / Divorced                                   |
| `NumberOfAddress`             | Delivery addresses registered                                 |
| `Complain`                    | Whether the customer filed a complaint **last month** (0 / 1) |
| `OrderAmountHikeFromlastYear` | Year-over-year % increase in order spend                      |
| `CouponUsed`                  | Number of coupons redeemed last month                         |
| `OrderCount`                  | Orders placed last month                                      |
| `DaySinceLastOrder`           | Days elapsed since last order                                 |
| `CashbackAmount`              | Average cashback received                                     |

### Class Imbalance

The dataset is imbalanced: **~83% retained** customers vs **~17% churned** customers. This is realistic and important — a naive model that always predicts "no churn" achieves 83% accuracy while being completely useless. Model evaluation therefore focuses on **recall** (catching actual churners) and **AUC-ROC** (discriminative ability across thresholds) rather than raw accuracy.

---

## Missing Values

Several features contain missing values (4–6% of records):

| Feature                     | Missing Count | Assessment                       |
|-----------------------------|---------------|----------------------------------|
| Tenure                      | 264           | MAR — imputed with MICE / median |
| WarehouseToHome             | 251           | MAR — imputed                    |
| HourSpendOnApp              | 255           | MAR — imputed                    |
| OrderAmountHikeFromlastYear | 265           | MAR — imputed                    |
| CouponUsed                  | 256           | MAR — imputed                    |
| OrderCount                  | 258           | MAR — imputed                    |
| DaySinceLastOrder           | 307           | MAR — imputed                    |

All missing patterns were assessed as **Missing At Random (MAR)** — no systematic relationship between missingness and the target was found — making median and iterative imputation appropriate.

---

## Business Objectives

The model should be evaluated against these business priorities in order:

1. **Maximise recall on churners** — a missed churner (false negative) costs more than a false alarm (false positive). The retention team would rather call a loyal customer unnecessarily than miss a genuinely at-risk one.

2. **Maintain acceptable precision** — too many false alarms overloads the retention team and devalues the intervention (customers receiving unnecessary discount offers is also a cost).

3. **Interpretability** — the model must produce feature importances that the marketing and product teams can act on. Knowing *why* a customer is predicted to churn is as valuable as the prediction itself.

4. **Production reliability** — the system must maintain stable performance as new monthly data arrives. A model that overfits to historical patterns will degrade rapidly in production.

---

## Two Complementary Approaches

This project addresses the business problem from two angles:

**Classification (Random Forest)** answers: *"Will this customer churn next month?"* — a binary decision useful for triggering real-time retention workflows.

**Survival Analysis (Cox Proportional Hazard Model)** answers: *"How long is this customer likely to stay, and at what rate is their hazard increasing?"* — a richer view useful for lifetime value forecasting and cohort-level strategy.

Together they give the business both an operational alert system and a strategic understanding of customer lifecycle dynamics.
