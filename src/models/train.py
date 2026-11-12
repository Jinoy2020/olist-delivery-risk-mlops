
from pathlib import Path
import json
import math

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_delayed"

EXCLUDED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "seller_states",
    "unique_sellers",
]


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        if TARGET not in df.columns:
            raise ValueError(f"{TARGET} missing from {name} dataset")

        if df[TARGET].isna().any():
            raise ValueError(f"Missing target values in {name} dataset")

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError(f"Unexpected target values in {name} dataset")

        if df.empty:
            raise ValueError(f"{name} dataset is empty")

    return train_df, val_df


# --------------------------------------------------
# Prepare features and target
# --------------------------------------------------

def prepare_features(train_df, val_df):

    columns_to_drop = [
        col for col in EXCLUDED_COLUMNS
        if col in train_df.columns
    ]

    X_train = train_df.drop(
        columns=[TARGET] + columns_to_drop
    )

    y_train = train_df[TARGET]

    X_val = val_df.drop(
        columns=[TARGET] + columns_to_drop,
        errors="ignore",
    )

    y_val = val_df[TARGET]

    # Keep validation columns identical to training columns.
    X_val = X_val.reindex(columns=X_train.columns)

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if (
        len(categorical_features) + len(numerical_features)
        != X_train.shape[1]
    ):
        raise ValueError("Unsupported feature data type detected")

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        categorical_features,
        numerical_features,
    )


# --------------------------------------------------
# Build preprocessing pipeline
# --------------------------------------------------

def build_preprocessor(
    categorical_features,
    numerical_features,
):

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ])


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

def evaluate_model(name, model, X_val, y_val):

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": name,

        "pr_auc": average_precision_score(
            y_val,
            probabilities,
        ),

        "roc_auc": roc_auc_score(
            y_val,
            probabilities,
        ),

        "precision": precision_score(
            y_val,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_val,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_val,
            predictions,
            zero_division=0,
        ),
    }

    return metrics


# --------------------------------------------------
# Model validation gate
# --------------------------------------------------


def validate_model_for_registration(metrics, y_val):
    """
    Validate the selected model before MLflow registration.

    Requirements:
    1. Evaluation metrics must be finite.
    2. PR-AUC must exceed positive-class prevalence.
    3. ROC-AUC must exceed 0.5.
    """

    def safe_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    # Safely calculate positive-class prevalence.
    try:
        baseline_pr_auc = safe_float(
            pd.Series(y_val).mean()
        )
    except (TypeError, ValueError):
        baseline_pr_auc = float("nan")

    # Safely retrieve and convert evaluation metrics.
    if not isinstance(metrics, dict):
        metrics = {}

    pr_auc = safe_float(metrics.get("pr_auc"))
    roc_auc = safe_float(metrics.get("roc_auc"))

    reasons = []

    values = [
        baseline_pr_auc,
        pr_auc,
        roc_auc,
    ]

    if not all(math.isfinite(value) for value in values):
        reasons.append(
            "Evaluation metrics contain missing, invalid, or non-finite values."
        )

    if math.isfinite(baseline_pr_auc):
        if not 0 <= baseline_pr_auc <= 1:
            reasons.append(
                "Positive-class baseline is outside the valid range."
            )

    if math.isfinite(pr_auc):
        if not 0 <= pr_auc <= 1:
            reasons.append(
                "PR-AUC is outside the valid range."
            )
        elif math.isfinite(baseline_pr_auc):
            if pr_auc <= baseline_pr_auc:
                reasons.append(
                    "PR-AUC does not exceed the positive-class baseline."
                )

    if math.isfinite(roc_auc):
        if not 0 <= roc_auc <= 1:
            reasons.append(
                "ROC-AUC is outside the valid range."
            )
        elif roc_auc <= 0.5:
            reasons.append(
                "ROC-AUC does not exceed random classification."
            )

    passed = len(reasons) == 0

    result = {
        "passed": passed,
        "baseline_pr_auc": (
            baseline_pr_auc
            if math.isfinite(baseline_pr_auc)
            else None
        ),
        "pr_auc": (
            pr_auc
            if math.isfinite(pr_auc)
            else None
        ),
        "roc_auc": (
            roc_auc
            if math.isfinite(roc_auc)
            else None
        ),
        "reasons": reasons,
    }

    if passed:
        print("\nMODEL VALIDATION PASSED")
    else:
        print("\nMODEL VALIDATION FAILED")

        for reason in reasons:
            print(f"- {reason}")

    return result


def log_validation_metadata(run_id, validation_result):
    """
    Log validation-gate results to the selected MLflow run.
    """

    with mlflow.start_run(run_id=run_id):

        mlflow.log_params({
            "validation_gate_passed": str(
                validation_result["passed"]
            ),
            "validation_gate_rule": (
                "pr_auc_above_baseline_and_roc_auc_above_0.5"
            ),
        })

        baseline_pr_auc = validation_result["baseline_pr_auc"]

        if baseline_pr_auc is not None:
            mlflow.log_metrics({
                "validation_baseline_pr_auc": baseline_pr_auc,
            })

        mlflow.set_tag(
            "validation_gate_status",
            "passed" if validation_result["passed"] else "failed",
        )

# --------------------------------------------------
# MLflow model registration
# --------------------------------------------------

def register_best_model(
    model_id,
    model_name="olist-delivery-risk-classifier",
):
    """
    Register the selected MLflow model.
    Does not promote it to production.
    """

    model_uri = f"models:/{model_id}"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(
        f"Registered model: {registered_model.name}, "
        f"version: {registered_model.version}"
    )

    return registered_model


def register_model_if_valid(
    model_id,
    metrics,
    y_val,
):
    """
    Register a model only if it passes validation.
    """

    validation_result = validate_model_for_registration(
        metrics,
        y_val,
    )

    if not validation_result["passed"]:
        raise ValueError(
            "Model registration blocked: "
            "the selected model failed validation."
        )

    registered_model = register_best_model(model_id)

    return registered_model, validation_result


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main(model_dir=None, register_model=False):

    model_dir = (
        Path(model_dir)
        if model_dir is not None
        else MODEL_DIR
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        "olist-delivery-risk"
    )

    print("Loading datasets...")

    train_df, val_df = load_data()

    (
        X_train,
        X_val,
        y_train,
        y_val,
        categorical_features,
        numerical_features,
    ) = prepare_features(
        train_df,
        val_df,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    print(
        f"Numerical features: {numerical_features}"
    )

    print(
        f"Categorical features: {categorical_features}"
    )

    # --------------------------------------------------
    # Define candidate models
    # --------------------------------------------------

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    trained_models = {}
    trained_model_ids = {}
    trained_run_ids = {}

    # --------------------------------------------------
    # Train and evaluate candidate models
    # --------------------------------------------------

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):

            active_run = mlflow.active_run()

            trained_run_ids[name] = (
                active_run.info.run_id
            )

            pipeline = Pipeline([
                (
                    "preprocessor",
                    build_preprocessor(
                        categorical_features,
                        numerical_features,
                    ),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ])

            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = evaluate_model(
                name,
                pipeline,
                X_val,
                y_val,
            )

            # Log model parameters.
            mlflow.log_params({
                "model_name": name,
                "class_weight": "balanced",
                "random_state": 42,
                "numerical_features": str(
                    numerical_features
                ),
                "categorical_features": str(
                    categorical_features
                ),
            })

            if name == "logistic_regression":

                mlflow.log_param(
                    "max_iter",
                    1000,
                )

            elif name == "random_forest":

                mlflow.log_param(
                    "n_estimators",
                    200,
                )

                mlflow.log_param(
                    "n_jobs",
                    -1,
                )

            # Log validation metrics.
            mlflow.log_metrics({
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            })

            # Log trained pipeline to MLflow.
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name=f"{name}_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            trained_model_ids[name] = (
                model_info.model_id
            )

        results.append(metrics)

        trained_models[name] = pipeline

        print(
            f"Validation metrics: {metrics}"
        )

    # --------------------------------------------------
    # Compare candidate models
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    ).sort_values(
        by="pr_auc",
        ascending=False,
    )

    print("\nModel comparison:")

    print(
        results_df.to_string(index=False)
    )

    best_name = results_df.iloc[0]["model"]

    best_model = trained_models[best_name]

    best_model_id = trained_model_ids[best_name]

    best_run_id = trained_run_ids[best_name]

    best_metrics = results_df.iloc[0].to_dict()

    print(f"\nSelected model: {best_name}")

    print(
        f"Selected MLflow model ID: {best_model_id}"
    )

    print(
        f"Selected MLflow run ID: {best_run_id}"
    )

    # --------------------------------------------------
    # Validate selected model
    # --------------------------------------------------

    validation_result = validate_model_for_registration(
        best_metrics,
        y_val,
    )

    # --------------------------------------------------
    # Log validation metadata to selected MLflow run
    # --------------------------------------------------

    log_validation_metadata(
    best_run_id,
    validation_result,
)

    # --------------------------------------------------
    # Save selected model and metrics
    # --------------------------------------------------

    model_path = (
        model_dir / "best_model.joblib"
    )

    metrics_path = (
        model_dir / "validation_metrics.json"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "best_model": best_name,
                "validation_metrics": results,
                "validation": validation_result,
                "mlflow_run_id": best_run_id,
            },
            file,
            indent=4,
        )

    print(f"\nBest model: {best_name}")

    print(f"Saved model: {model_path}")

    print(f"Saved metrics: {metrics_path}")

    # --------------------------------------------------
    # Register only after successful validation
    # --------------------------------------------------

    registered_model = None

    if register_model:

        if not validation_result["passed"]:

            raise ValueError(
                "Model registration blocked: "
                "the selected model failed validation."
            )

        registered_model = register_best_model(
            best_model_id
        )

    # --------------------------------------------------
    # Return training results
    # --------------------------------------------------

    return {
        "best_model": best_name,

        "model_id": best_model_id,

        "mlflow_run_id": best_run_id,

        "validation": validation_result,

        "registered_model": (
            {
                "name": registered_model.name,
                "version": registered_model.version,
            }
            if registered_model is not None
            else None
        ),
    }


# --------------------------------------------------
# Script entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()
