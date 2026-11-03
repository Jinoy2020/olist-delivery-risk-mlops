import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.train import (
    load_data,
    prepare_features,
    build_preprocessor,
)


def run_experiment(name, X_train, X_val, y_train, y_val):
    categorical = X_train.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numerical = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    preprocessor = build_preprocessor(categorical, numerical)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    return {
        "experiment": name,
        "PR-AUC": average_precision_score(y_val, probabilities),
        "ROC-AUC": roc_auc_score(y_val, probabilities),
    }


def main():
    train_df, val_df = load_data()

    (
        X_train,
        X_val,
        y_train,
        y_val,
        categorical,
        numerical,
    ) = prepare_features(train_df, val_df)

    if "estimated_lead_days" not in X_train.columns:
        raise ValueError("estimated_lead_days not found")

    results = []

    # Model A: With estimated lead time
    results.append(run_experiment(
        "With estimated_lead_days",
        X_train.copy(),
        X_val.copy(),
        y_train,
        y_val,
    ))

    # Model B: Without estimated lead time
    results.append(run_experiment(
        "Without estimated_lead_days",
        X_train.drop(columns=["estimated_lead_days"]),
        X_val.drop(columns=["estimated_lead_days"]),
        y_train,
        y_val,
    ))

    results_df = pd.DataFrame(results)

    print("\nFeature ablation results:")
    print(results_df.to_string(index=False))

    print("\nPR-AUC improvement:")
    print(
        results_df.iloc[0]["PR-AUC"]
        - results_df.iloc[1]["PR-AUC"]
    )


if __name__ == "__main__":
    main()