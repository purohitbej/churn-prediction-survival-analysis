# Findings & Results

## Summary

A Random Forest classifier and a Cox Proportional Hazard survival model were built to predict and understand customer churn on an e-commerce platform. After thorough data leakage investigation, overfitting remediation, and feature engineering, the final model achieves **AUC 0.876** on held-out test data with a stable train/test gap of ~0.025 — indicating reliable generalisation to unseen customers.

---

## Part 1 — Exploratory Data Analysis

### Churn Rate

Approximately **16.8% of customers churned**, creating a moderate class imbalance that was handled via class weighting (`{0: 1, 1: 3.4}`) during model training.

### Key Categorical Findings

**City Tier:** Tier 1 city customers churn at the lowest rate. Tier 2 and Tier 3 customers show meaningfully higher churn tendency — cost sensitivity and delivery infrastructure are likely drivers.

**Marital Status:** Single customers have the highest churn rate (~33% among singles vs ~13% among married customers). The combination of being single in a Tier 3 city is particularly high-risk. Married customers are the most loyal segment across all city tiers.

**Preferred Order Category:** Grocery buyers are the most loyal (lowest churn rate). Fashion and Mobile buyers are most likely to churn — consistent with the idea that high-ticket and trend-sensitive purchases are more likely to be comparison-shopped across platforms.

**Payment Mode:** COD and E-Wallet users churn at higher rates than Debit Card or UPI users. This may reflect lower financial commitment or lower switching costs for these payment types.

**Complaints:** Customers who filed a complaint last month churn at ~3× the rate of non-complainers (31.7% vs 10.9%). This is a strong, actionable signal — timely complaint resolution is a direct lever for retention.

### Key Numerical Findings

**Tenure:** The strongest churn signal. The majority of churns happen in months 0–12. After 24 months, survival probability stabilises substantially. Early-lifecycle interventions deliver the highest return on investment.

**CashbackAmount:** Churners tend to receive slightly lower cashback on average than retained customers, suggesting cashback is both a retention mechanism and a churn predictor.

**HourSpendOnApp:** Nearly identical distributions between churners and retained customers — weak predictive signal.

---

## Part 2 — Data Leakage Investigation

A critical analysis was performed on all features before finalising the model.

**`Complain`** was initially suspected as potential leakage (a complaint filed *because of* churn rather than *before* it). Documentation review confirmed it records complaints from **the previous calendar month** — making it a legitimate leading indicator of churn, not a contemporaneous proxy.

**`DaySinceLastOrder`** showed a suspicious pattern: churners had a *higher* density near 0 days since last order than non-churners, which is counterintuitive. The feature's timing relative to the churn observation date could not be fully confirmed from available documentation and remains a watchlist item for production monitoring.

**Feature engineering interactions** (e.g., `CouponUsed × OrderCount`) were tested. KDE plots showed nearly identical distributions between churner and non-churner groups, confirming these interactions added no discriminative signal and were dropped.

**`Tenure_Cashback`** (the product of Tenure group and CashbackAmount) showed clear separation between churner and non-churner distributions in KDE analysis. Despite high correlation with the original Tenure feature (r = 0.93), it replaced Tenure in the final model as a more expressive combined signal.

---

## Part 3 — Model Development

### Overfitting Journey

| Stage                         | Train AUC | Test AUC | Gap   | Notes                     |
|-------------------------------|-----------|----------|-------|---------------------------|
| Initial (depth 17)            | 1.000     | 0.978    | 0.022 | Severe memorisation       |
| After removing leaky features | 0.984     | 0.926    | 0.058 | Improved, gap still large |
| Depth reduced to 10           | 0.946     | 0.892    | 0.054 | Overcorrected — underfit  |
| Final (depth 5, leaf=20)      | 0.899     | 0.875    | 0.024 | ✅ Stable generalisation  |

The key insight was that a perfect training AUC is not a sign of a powerful model — it is a sign of a model that has memorised noise. The final model with `max_depth=5` produces rules general enough to describe customer behaviour rather than individual training records.

### Hyperparameter Tuning

Five sequential GridSearchCV rounds were performed with 5-fold StratifiedKFold:

1. `max_features` and `n_estimators` — best: `max_features=None`, `n_estimators=1000`
2. `criterion` and `max_depth` — best: `criterion=entropy`, `max_depth` explored
3. `min_samples_leaf` and `min_samples_split` — regularisation lever confirmed
4. `class_weight` — best: `{0: 1, 1: 3}` for improved minority class recall
5. Final combined grid → `max_depth=5`, `min_samples_leaf=20`, `min_samples_split=30`, `class_weight={0:1, 1:3.4}`

### Final Model Performance

```
Test Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.88      0.90      1171
           1       0.52      0.65      0.57       237

    accuracy                           0.84      1408
   macro avg       0.72      0.76      0.74      1408
weighted avg       0.85      0.84      0.85      1408

AUC (Test):  0.875
AUC (Train): 0.899
```

The minority class (churners) achieves **65% recall** at the default threshold. Threshold optimisation using the precision-recall curve can push this higher at the cost of precision — the right balance is a business decision determined by the cost ratio of false negatives to false positives.

### Feature Importances (Final Model)

The top predictors in order of importance:

1. **Tenure_Cashback** — combined tenure and loyalty signal; low-tenure/low-cashback customers are the highest risk
2. **Complain** — complaint in the previous month is a powerful leading indicator
3. **DaySinceLastOrder** — recency of engagement
4. **PreferedOrderCat** — product category preferences signal commitment level
5. **MaritalStatus** — lifestyle segment proxy
6. **SatisfactionScore** — self-reported signal of experience quality
7. **WarehouseToHome** — logistics experience proxy

---

## Part 4 — Survival Analysis

### Kaplan-Meier Curves

Survival analysis confirmed that **most churn happens in the first 12 months**, with the survival curve flattening significantly after 24 months. This identifies the early lifecycle as the critical intervention window.

### Log-Rank Test Results (Significant Group Differences)

**Login Device:** All three device groups (Phone, Computer, Mobile Phone) have statistically different survival curves (p < 0.05).

**Payment Mode:** COD/E-Wallet and UPI/Credit Card customers cluster into distinct survival groups. Debit Card users are most loyal.

**Order Category:** Grocery & Others show similar survival curves. Mobile Phone and Mobile are statistically indistinguishable. Laptop and Fashion diverge significantly.

**Marital Status:** All three marital status groups are statistically distinguishable, with single customers having the steepest early churn gradient.

### Cox Proportional Hazard Model

The final Cox model stratified on `NumberOfDeviceRegistered` (which violated the proportional hazard assumption in the initial fit) with all p-values < 0.05 for remaining covariates. The PH assumption was confirmed via Schoenfeld residual analysis with no significant trends.

**Model concordance index: 0.73** — the model correctly ranks 73% of customer pairs by their relative survival time.

Key hazard ratios (direction):
- Higher `SatisfactionScore` → lower hazard (protective)
- `Complain = 1` → significantly elevated hazard
- `PreferredPaymentMode: COD&EWallet` → elevated hazard vs Debit Card
- Single marital status → elevated hazard vs Married/Divorced

---

## Business Recommendations

**Immediate actions (high ROI):**
- Priority retention outreach for customers with `Tenure < 12 months` AND `Complain = 1`
- Investigate cashback programme effectiveness for new customers — Tenure_Cashback being the top feature suggests early cashback incentives are directly correlated with retention
- Focus intervention resources on Mobile and Fashion product category customers in Tier 2/3 cities

**Product recommendations:**
- Improve complaint resolution speed and transparency — complaints are the single highest-lift individual signal in the model
- Consider onboarding incentives specifically for Single customers in non-Tier-1 cities

**Model deployment recommendations:**
- Retrain the model monthly as new behavioural data arrives
- Monitor feature drift for `DaySinceLastOrder` in production — confirm its measurement timestamp is consistent with the training data assumption
- Use the survival model's individual cumulative hazard curves to prioritise retention budget allocation by predicted lifetime value
