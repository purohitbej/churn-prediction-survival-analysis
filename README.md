# 🛒 E-Commerce Customer Churn Prediction

> End-to-end machine learning system to predict, understand, and act on customer churn — combining a Random Forest classifier with a Cox Proportional Hazard survival model, served via a production-ready Flask API.

---

## 📌 Overview

Customer churn is one of the most expensive problems in e-commerce. This project builds a complete, production-grade ML system to:

- **Predict** which customers are likely to churn next month (Random Forest, AUC = 0.875)
- **Understand** *when* and *why* customers leave (Cox Proportional Hazard model, C-index = 0.73)
- **Serve predictions** through a REST API deployable anywhere

See [`docs/BUSINESS_PROBLEM.md`](docs/BUSINESS_PROBLEM.md) for full context and [`docs/FINDINGS_AND_RESULTS.md`](docs/FINDINGS_AND_RESULTS.md) for a complete writeup of findings.

---

## 🗂️ Repository Structure

```
ecommerce-churn-prediction/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Data profiling, categorical & numerical analysis
│   ├── 02_feature_engineering.ipynb         # Interaction terms, correlation analysis, EDA conclusions
│   ├── 03_model_building.ipynb              # Pipeline, train/test evaluation, churn_prediction()
│   ├── 04_hyperparameter_tuning.ipynb       # 5 sequential GridSearchCV rounds
│   └── 05_survival_analysis.ipynb           # Kaplan-Meier, log-rank tests, Cox PH model
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py   # Ingestion, cleaning, encoding, Tenure_Cashback feature
│   ├── model.py              # Pipeline factory, churn_prediction(), threshold optimisation
│   └── survival.py           # KM curves, log-rank tests, Cox PH fitting & diagnostics
│
├── api/
│   ├── __init__.py
│   ├── app.py                # Flask app — /health, /predict, /predict/batch
│   ├── schemas.py            # Pydantic request/response models
│   └── utils.py              # Model loading, feature prep, risk tiering
│
├── scripts/
│   └── train.py              # End-to-end training script — dumps model artefacts
│
├── models/                   # Saved artefacts (created by train.py, excluded from git)
│   ├── churn_pipeline.joblib
│   ├── encoder.joblib
│   ├── threshold.txt
│   └── feature_names.txt
│
├── tests/
│   └── test_api.py           # Pytest suite for API endpoints
│
├── docs/
│   ├── BUSINESS_PROBLEM.md
│   └── FINDINGS_AND_RESULTS.md
│
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction
pip install -r requirements.txt
```

### 2. Train the Model

Place your dataset in a `data/` folder, then run:

```bash
python scripts/train.py --data "data/E Commerce Dataset.xlsx" --excel --sheet "E Comm"
```

This saves trained artefacts to `models/`.

### 3. Start the API

```bash
python wsgi.py
```

The API runs at `http://localhost:5000`.

### 4. Make a Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PreferredLoginDevice": "Phone",
    "CityTier": 3,
    "WarehouseToHome": 25.0,
    "PreferredPaymentMode": "COD",
    "Gender": "Male",
    "HourSpendOnApp": 2.0,
    "NumberOfDeviceRegistered": 4,
    "PreferedOrderCat": "Mobile",
    "SatisfactionScore": 2,
    "MaritalStatus": "Single",
    "NumberOfAddress": 3,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 12.0,
    "CouponUsed": 1.0,
    "OrderCount": 2.0,
    "DaySinceLastOrder": 3.0,
    "Tenure": 4.0,
    "CashbackAmount": 150.0
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7841,
  "churn_prediction": 1,
  "threshold_used": 0.4123,
  "risk_level": "High"
}
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| AUC-ROC (Test) | 0.875 |
| AUC-ROC (Train) | 0.899 |
| Train/Test Gap | 0.024 ✅ |
| Churn Recall | 0.65 |
| Churn F1 | 0.57 |
| Concordance Index (Cox) | 0.73 |

The small train/test AUC gap (0.024) confirms the model generalises well and is not memorising the training set.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch predictions |

### Risk Tiers

| Probability | Risk Level |
|---|---|
| < 0.30 | 🟢 Low |
| 0.30 – 0.55 | 🟡 Medium |
| 0.55 – 0.75 | 🟠 High |
| > 0.75 | 🔴 Critical |

---

## 🚢 Deployment

### Local / Development

```bash
python wsgi.py
```

### Production (Gunicorn)

Gunicorn is a production-grade WSGI server — more stable and performant than Flask's built-in dev server.

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

`-w 4` spawns 4 worker processes. A good rule of thumb is `2 × CPU cores + 1`.

### Deploying to a Cloud VM (e.g. AWS EC2, GCP Compute Engine)

```bash
# 1. SSH into your server and clone the repo
git clone https://github.com/yourusername/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Upload your dataset and train the model
python scripts/train.py --data "data/E Commerce Dataset.xlsx" --excel --sheet "E Comm"

# 4. Start the server
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

The API will be accessible at `http://<your-server-ip>:5000`.

> **Note:** For a public-facing deployment, put Nginx in front of Gunicorn to handle HTTPS and rate limiting. For quick demos, the above is sufficient.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📖 Notebooks Guide

| Notebook | What to read it for |
|---|---|
| `01_exploratory_data_analysis` | Understanding the data, class distributions, missing value patterns |
| `02_feature_engineering` | Interaction term analysis, Pearson correlations, feature decisions |
| `03_model_building` | Pipeline setup, evaluation function, final model performance |
| `04_hyperparameter_tuning` | GridSearchCV results, overfitting journey, parameter decisions |
| `05_survival_analysis` | Kaplan-Meier curves, log-rank tests, Cox PH fitting and diagnostics |

---

## 🏗️ Key Technical Decisions

**Why Random Forest over SVM?** SVMs don't output calibrated probabilities natively, making threshold tuning unreliable. Random Forests produce reliable probability estimates and handle tabular imbalanced data well.

**Why `max_depth=5`?** Extensive experimentation showed that deeper trees (up to depth 17) memorised training data perfectly (AUC = 1.0 on train) but showed a ~0.05+ gap on test. Depth 5 reduces this gap to 0.024 while maintaining strong AUC.

**Why the `Tenure_Cashback` interaction?** KDE analysis showed clear distribution separation between churners and non-churners — something neither raw Tenure nor CashbackAmount achieved alone. The product encodes the compound effect of loyalty × spend incentive.

**Why survival analysis in addition to classification?** Classification tells you *who*. Survival analysis tells you *when* and at *what rate*. Both are needed for a complete retention strategy — particularly for lifetime value calculations and budget allocation.

---

## 📄 License

MIT
