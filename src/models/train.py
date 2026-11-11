
from pathlib import Path
import json

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
    classification_report,
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

# Exclude identifiers, timestamps and seller-derived
# features pending checkout-time availability verification.
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

    for name, df in [("train", train_df), ("validation", val_df)]:
        if TARGET not in df.columns:
            raise ValueError(f"{TARGET} missing from {name} dataset")

        if df[TARGET].isna().any():
            raise ValueError(f"Missing target values in {name} dataset")

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError(f"Unexpected target values in {name} dataset")

    return train_df, val_df


# --------------------------------------------------
# Prepare features and target
# --------------------------------------------------

def prepare_features(train_df, val_df):
    columns_to_drop = [
        col for col in EXCLUDED_COLUMNS
        if col in train_df.columns
    ]

    X_train = train_df.drop(columns=[TARGET] + columns_to_drop)
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=[TARGET] + columns_to_drop)
    y_val = val_df[TARGET]

    # Keep validation columns identical to training columns.
    X_val = X_val.reindex(columns=X_train.columns)

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    if len(categorical_features) + len(numerical_features) != X_train.shape[1]:
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

def build_preprocessor(categorical_features, numerical_features):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


# --------------------------------------------------
# Train and evaluate models
# --------------------------------------------------

def evaluate_model(name, model, X_val, y_val):
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "pr_auc": average_precision_score(y_val, probabilities),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
    }

    return metrics


def main(model_dir=None):
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("olist-delivery-risk")

    print("Loading datasets...")
    train_df, val_df = load_data()

    (
        X_train,
        X_val,
        y_train,
        y_val,
        categorical_features,
        numerical_features,
    ) = prepare_features(train_df, val_df)

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    preprocessor = build_preprocessor(
        categorical_features,
        numerical_features,
    )

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

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor(
                    categorical_features,
                    numerical_features,
                )),
                ("classifier", classifier),
            ])

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                name,
                pipeline,
                X_val,
                y_val,
            )

            # Log model parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("numerical_features", numerical_features)
            mlflow.log_param("categorical_features", categorical_features)

            if name == "logistic_regression":
                mlflow.log_param("max_iter", 1000)

            elif name == "random_forest":
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("n_jobs", -1)

            # Log validation metrics
            mlflow.log_metrics({
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            })

        
                        # Log trained pipeline
            mlflow.sklearn.log_model(
                pipeline,
                name=f"{name}_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

        print(f"Validation metrics: {metrics}")
        results.append(metrics)
        trained_models[name] = pipeline

        print(f"Validation metrics: {metrics}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    model_path = model_dir / "best_model.joblib"
    metrics_path = model_dir / "validation_metrics.json"

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_name,
                "validation_metrics": results,
            },
            file,
            indent=4,
        )

    print(f"\nBest model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
