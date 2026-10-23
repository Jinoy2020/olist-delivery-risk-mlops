
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_COLUMNS = {
    "olist_orders_dataset.csv": [
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ],
    "olist_order_items_dataset.csv": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
    ],
    "olist_customers_dataset.csv": [
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
    ],
    "olist_sellers_dataset.csv": [
        "seller_id",
        "seller_city",
        "seller_state",
    ],
}


def validate_file(filename: str, required_columns: list[str]) -> bool:
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, nrows=5)

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        print(f"FAIL: {filename} is missing columns: {sorted(missing_columns)}")
        return False

    print(f"PASS: {filename} contains all required columns.")
    return True


def main() -> None:
    all_valid = True

    for filename, columns in REQUIRED_COLUMNS.items():
        if not validate_file(filename, columns):
            all_valid = False

    if not all_valid:
        raise SystemExit("Data validation failed.")

    print("\nAll datasets passed column validation!")


if __name__ == "__main__":
    main()
