"""
train.py
--------
End-to-end training script.

Usage
-----
    python scripts/train.py --data "data/E Commerce Dataset.xlsx" --excel --sheet "E Comm"
    python scripts/train.py --data data/ecommerce.csv

Outputs
-------
    models/churn_pipeline.joblib   — the fitted sklearn Pipeline
    models/encoder.joblib          — the fitted OrdinalEncoder
    models/threshold.txt           — the optimal decision threshold
    models/feature_names.txt       — ordered list of feature columns
"""

import argparse
import os
import joblib
import mlflow

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
)

# Make sure the project root is on the path when running as a script
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preparation import load_and_prepare, get_feature_target_split
from src.model import build_pipeline_xgb, get_optimal_threshold, BEST_PARAMS_XGB


def parse_args():
    parser = argparse.ArgumentParser(description="Train the churn prediction model.")
    parser.add_argument("--data", required=True, help="Path to the dataset file.")
    parser.add_argument(
        "--excel", action="store_true", help="Set if file is an Excel workbook."
    )
    parser.add_argument(
        "--sheet", default=None, help="Excel sheet name (required if --excel)."
    )
    parser.add_argument(
        "--output-dir", default="models", help="Directory to save artefacts."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Test split fraction."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for train/test split."
    )
    parser.add_argument(
        "--experiment", default="churn-prediction", help="MLflow experiment name."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load & prepare data
    # ======================
    print("Loading and preparing data...")
    df, encoder = load_and_prepare(
        file_path=args.data,
        is_excel=args.excel,
        sheet_name=args.sheet,
        fit_encoder=True,
    )

    X, y = get_feature_target_split(df)
    feature_names = X.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        print("Training XGBoost pipeline ...")

        mlflow.log_params(dict({"model_type": "XGBoost"}, **BEST_PARAMS_XGB))

        # 2. Cross-validation on training data
        # ====================================
        print("\nRunning 5-fold cross-validation (AUC)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipeline = build_pipeline_xgb()
        cv_scores = cross_val_score(
            pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 3. Final training on full training set

        print("\nTraining final model on full training set...")
        final_pipeline = build_pipeline_xgb()
        final_pipeline.fit(x_train, y_train)

        # 4. Threshold optimisation
        # =========================
        print("\nOptimising decision threshold on test set...")
        threshold = get_optimal_threshold(final_pipeline, x_test, y_test)

        # 5. Evaluation
        # =============
        test_probs = final_pipeline.predict_proba(x_test)[:, 1]
        train_probs = final_pipeline.predict_proba(x_train)[:, 1]
        test_preds = (test_probs >= threshold).astype(int)
        train_preds = (train_probs >= threshold).astype(int)

        test_auc = roc_auc_score(y_test, test_probs)
        train_auc = roc_auc_score(y_train, train_probs)
        test_f1 = f1_score(y_test, test_preds)
        train_f1 = f1_score(y_train, train_preds)
        test_recall = recall_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)

        print(f"Test  AUC: {test_auc:.4f}  |  Train AUC: {train_auc:.4f}")
        print(f"Test  F1 : {test_f1:.4f}  |  Train F1 : {train_f1:.4f}")
        print(f"Test  Recall: {test_recall:.4f}")
        print(f"Test  Precision: {test_precision:.4f}")
        print("\nFull test classification report:")
        print(classification_report(y_test, test_preds))

        # Log metrics for comparison
        mlflow.log_metrics(
            {
                "test_auc": test_auc,
                "train_auc": train_auc,
                "auc_gap": train_auc - test_auc,
                "test_f1": test_f1,
                "train_f1": train_f1,
                "f1_gap": train_f1 - test_f1,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "optimal_threshold": threshold,
            }
        )

        # 6. Save artefacts
        # =================
        pipeline_path = os.path.join(args.output_dir, "churn_pipeline.joblib")
        encoder_path = os.path.join(args.output_dir, "encoder.joblib")
        threshold_path = os.path.join(args.output_dir, "threshold.txt")
        features_path = os.path.join(args.output_dir, "feature_names.txt")

        joblib.dump(final_pipeline, pipeline_path)
        joblib.dump(encoder, encoder_path)

        with open(threshold_path, "w") as f:
            f.write(str(threshold))

        with open(features_path, "w") as f:
            f.write("\n".join(feature_names))

        print(f"\nArtefacts saved to '{args.output_dir}/':")
        print(f"  {pipeline_path}")
        print(f"  {encoder_path}")
        print(f"  {threshold_path}")
        print(f"  {features_path}")

        mlflow.log_artifact(pipeline_path)
        mlflow.log_artifact(threshold_path)
        mlflow.log_artifact(features_path)

        mlflow.sklearn.log_model(
            pipeline, artifact_path="model", registered_model_name="churn-xgb"
        )

        print("\nTraining complete. Run 'mlflow ui' to inspect this run.")


if __name__ == "__main__":
    main()
