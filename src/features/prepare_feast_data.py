
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "delivery_risk_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feast_order_features.parquet"
)

MODEL_FEATURES = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "estimated_lead_days",
    "item_count",
    "total_price",
    "total_freight_value",
    "customer_state",
]


def prepare_feast_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "order_id",
        "order_purchase_timestamp",
        *MODEL_FEATURES,
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feast_df = df[required_columns].copy()

    feast_df = feast_df.rename(
        columns={
            "order_purchase_timestamp": "event_timestamp"
        }
    )

    feast_df["event_timestamp"] = pd.to_datetime(
        feast_df["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    if feast_df["event_timestamp"].isna().any():
        raise ValueError("Invalid event timestamps detected.")

    if feast_df["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs detected.")
    if feast_df["order_id"].isna().any():
        raise ValueError("Missing order IDs detected.")
    if feast_df[MODEL_FEATURES].isna().any().any():
        raise ValueError("Missing model feature values detected.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feast_df.to_parquet(OUTPUT_PATH, index=False)

    print("Feast dataset created successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(feast_df):,}")
    print(f"Columns: {len(feast_df.columns)}")
    print("\nFeature columns:")
    print(", ".join(feast_df.columns))


if __name__ == "__main__":
    prepare_feast_data()
