"""
app.py
------
Flask REST API for the Customer Churn Prediction model.

Endpoints
---------
GET  /health                   — liveness check
POST /predict                  — predict churn for a single customer
POST /predict/batch            — predict churn for multiple customers

Running locally
---------------
    python -m api.app

Running with gunicorn (production)
-----------------------------------
    gunicorn -w 4 -b 0.0.0.0:5000 "api.app:create_app()"
"""

import os
import traceback

import numpy as np
from flask import Flask, request, jsonify

from api.utils import (
    load_pipeline,
    load_threshold,
    customer_to_dataframe,
    risk_level,
)


# App factory
# ===========
def create_app() -> Flask:
    app = Flask(__name__)

    # Eagerly load the model at startup so the first request isn't slow
    try:
        load_pipeline()
        load_threshold()
        app.logger.info("Model loaded successfully.")
    except FileNotFoundError as e:
        app.logger.warning(f"Model not pre-loaded: {e}")

    # Routes
    # ======
    @app.route("/health", methods=["GET"])
    def health():
        """Liveness check — returns 200 if the API is running."""
        try:
            pipeline = load_pipeline()
            return jsonify({"status": "ok", "model_loaded": True}), 200
        except FileNotFoundError:
            return jsonify({"status": "degraded", "model_loaded": False}), 503

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Predict churn probability for a single customer.

        Request body (JSON)
        -------------------
        See api/schemas.py → CustomerFeatures for all fields.

        Response (JSON)
        ---------------
        {
            "churn_probability": 0.73,
            "churn_prediction": 1,
            "threshold_used": 0.42,
            "risk_level": "High"
        }
        """
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body must be JSON."}), 400

            pipeline = load_pipeline()
            threshold = load_threshold()

            df = customer_to_dataframe(data)
            prob = float(pipeline.predict_proba(df)[:, 1][0])
            prediction = int(prob >= threshold)

            return (
                jsonify(
                    {
                        "churn_probability": round(prob, 4),
                        "churn_prediction": prediction,
                        "threshold_used": round(threshold, 4),
                        "risk_level": risk_level(prob),
                    }
                ),
                200,
            )

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 503
        except KeyError as e:
            return jsonify({"error": f"Missing required field: {e}"}), 422
        except Exception:
            return jsonify({"error": traceback.format_exc()}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """
        Predict churn for a list of customers.

        Request body (JSON)
        -------------------
        {
            "customers": [ { ...CustomerFeatures... }, ... ]
        }

        Response (JSON)
        ---------------
        {
            "predictions": [ { ...PredictionResponse... }, ... ],
            "total_customers": 5,
            "predicted_churners": 2,
            "churn_rate": 0.4
        }
        """
        try:
            data = request.get_json(force=True)
            if not data or "customers" not in data:
                return jsonify({"error": "Expected JSON with a 'customers' list."}), 400

            customers = data["customers"]
            if not isinstance(customers, list) or len(customers) == 0:
                return jsonify({"error": "'customers' must be a non-empty list."}), 400

            pipeline = load_pipeline()
            threshold = load_threshold()

            results = []
            for customer in customers:
                df = customer_to_dataframe(customer)
                prob = float(pipeline.predict_proba(df)[:, 1][0])
                prediction = int(prob >= threshold)
                results.append(
                    {
                        "churn_probability": round(prob, 4),
                        "churn_prediction": prediction,
                        "threshold_used": round(threshold, 4),
                        "risk_level": risk_level(prob),
                    }
                )

            total = len(results)
            churners = sum(r["churn_prediction"] for r in results)

            return (
                jsonify(
                    {
                        "predictions": results,
                        "total_customers": total,
                        "predicted_churners": churners,
                        "churn_rate": round(churners / total, 4) if total > 0 else 0.0,
                    }
                ),
                200,
            )

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 503
        except Exception:
            return jsonify({"error": traceback.format_exc()}), 500

    return app


# Dev server entrypoint
# =====================
if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
