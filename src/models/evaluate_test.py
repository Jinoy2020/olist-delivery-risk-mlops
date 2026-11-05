from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "test.csv"
REPORT_DIR = ROOT_DIR / "reports"
REPORT_PATH = REPORT_DIR / "test_evaluation.json"

MODEL_NAME = "olist-delivery-risk-classifier"
MODEL_VERSION = "1"
THRESHOLD = 0.50

TARGET = "is_delayed"

EXCLUDED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "seller_states",
    "unique_sellers",
]


# --------------------------------------------------
# Load test data
# --------------------------------------------------

def load_test_data():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError("Target column missing from test dataset.")

    if df[TARGET].isna().any():
        raise ValueError("Missing target values in test dataset.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Unexpected target values.")

    drop_columns = [
        col for col in EXCLUDED_COLUMNS
        if col in df.columns
    ]

    X_test = df.drop(columns=[TARGET] + drop_columns)
    y_test = df[TARGET].astype(int)

    return X_test, y_test


# --------------------------------------------------
# Load registered model
# --------------------------------------------------

def load_registered_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    return model


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

def evaluate():
    print("Loading held-out test dataset...")

    X_test, y_test = load_test_data()

    print("Test dataset shape:", X_test.shape)
    print("Actual class distribution:")
    print(y_test.value_counts().to_dict())

    print("\nLoading registered model...")

    model = load_registered_model()

    expected_features = model.feature_names_in_.tolist()

    if X_test.columns.tolist() != expected_features:
        raise ValueError(
            "Test features do not match registered model features."
        )

    print("Registered model loaded successfully.")

    # Generate predictions
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Calculate metrics
    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "test_samples": len(y_test),
        "actual_delayed": int(y_test.sum()),
        "actual_on_time": int((y_test == 0).sum()),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["On Time", "Delayed"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n========== TEST EVALUATION ==========")

    for key, value in metrics.items():
        if key != "classification_report":
            print(f"{key}: {value}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["On Time", "Delayed"],
            zero_division=0,
        )
    )

    print("\nTest evaluation saved to:", REPORT_PATH)


if __name__ == "__main__":
    evaluate()