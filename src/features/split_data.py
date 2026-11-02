from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_risk_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["order_purchase_timestamp"])

    required = {
        "order_id",
        "order_purchase_timestamp",
        "is_delayed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found.")

    if df["order_purchase_timestamp"].isna().any():
        raise ValueError("Missing purchase timestamps found.")

    # Stable chronological order for reproducibility.
    df = df.sort_values(
        ["order_purchase_timestamp", "order_id"]
    ).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        split.to_csv(output_path, index=False)

        print(
            f"{name}: {len(split):,} rows | "
            f"{split['order_purchase_timestamp'].min()} to "
            f"{split['order_purchase_timestamp'].max()} | "
            f"delayed rate={split['is_delayed'].mean():.3f}"
        )

    # Verify the chronological boundaries do not overlap.
    assert train["order_purchase_timestamp"].max() <= (
        validation["order_purchase_timestamp"].min()
    )
    assert validation["order_purchase_timestamp"].max() <= (
        test["order_purchase_timestamp"].min()
    )

    print("\nChronological split completed successfully.")


if __name__ == "__main__":
    main()