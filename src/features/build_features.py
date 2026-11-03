from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}


def load_csv(name: str) -> pd.DataFrame:
    """Load a required raw dataset."""
    path = RAW_DIR / FILES[name]

    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")

    return pd.read_csv(path)


def build_features() -> pd.DataFrame:
    """Build one leakage-aware training row per delivered order."""

    orders = load_csv("orders")
    items = load_csv("items")
    customers = load_csv("customers")
    sellers = load_csv("sellers")

    # Check required columns before processing.
    required_order_cols = {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    }
    missing = required_order_cols - set(orders.columns)
    if missing:
        raise ValueError(f"Orders dataset missing columns: {sorted(missing)}")

    required_item_cols = {
        "order_id", "order_item_id", "seller_id", "price", "freight_value"
    }
    missing = required_item_cols - set(items.columns)
    if missing:
        raise ValueError(f"Items dataset missing columns: {sorted(missing)}")

    # Parse timestamps. Actual delivery is used only to create the label.
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders["order_estimated_delivery_date"] = pd.to_datetime(
        orders["order_estimated_delivery_date"], errors="coerce"
    )
    actual_delivery = pd.to_datetime(
        orders["order_delivered_customer_date"], errors="coerce"
    )
        # Estimated delivery lead time, available at checkout (subject to validation).
    orders["estimated_lead_days"] = (
        orders["order_estimated_delivery_date"]
        - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    # Historical labels require a known outcome and a valid purchase time.
    valid = (
        orders["order_status"].eq("delivered")
        & orders["order_purchase_timestamp"].notna()
        & orders["order_estimated_delivery_date"].notna()
        & actual_delivery.notna()
    )

    labeled_orders = orders.loc[valid].copy()
    labeled_actual = actual_delivery.loc[valid]

    labeled_orders["is_delayed"] = (
        labeled_actual
        > labeled_orders["order_estimated_delivery_date"]
    ).astype("int8")

    # Purchase-time calendar features.
    purchase_time = labeled_orders["order_purchase_timestamp"]
    labeled_orders["purchase_month"] = purchase_time.dt.month
    labeled_orders["purchase_day_of_week"] = purchase_time.dt.day_of_week
    labeled_orders["purchase_hour"] = purchase_time.dt.hour

    # Item-level features aggregated to one row per order.
    item_features = items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        unique_sellers=("seller_id", "nunique"),
    ).reset_index()

    # Customer location known from the order's customer record.
    customer_features = customers[
        ["customer_id", "customer_state"]
    ].copy()

    # Seller location features based on sellers assigned to the order.
    seller_items = items[["order_id", "seller_id"]].merge(
        sellers[["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    seller_features = seller_items.groupby("order_id").agg(
        seller_states=("seller_state", "nunique"),
    ).reset_index()

    # Keep order ID and purchase timestamp for auditing and temporal splitting.
    dataset = labeled_orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "purchase_month",
            "purchase_day_of_week",
            "purchase_hour",
            "estimated_lead_days",
            "is_delayed",
        ]
    ].merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    dataset = dataset.merge(
        customer_features,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    dataset = dataset.merge(
        seller_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # Identifiers and timestamps are retained for audit/splitting,
    # but MUST be excluded from model features.
    if dataset["order_id"].duplicated().any():
        raise ValueError("Feature dataset contains duplicate order IDs.")

    if dataset["is_delayed"].isna().any():
        raise ValueError("Target contains missing values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "delivery_risk_features.csv"
    dataset.to_csv(output_path, index=False)

    print("Feature extraction completed.")
    print(f"Output: {output_path}")
    print(f"Rows: {len(dataset):,}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Unique orders: {dataset['order_id'].nunique():,}")
    print("\nTarget distribution:")
    print(dataset["is_delayed"].value_counts().sort_index())
    print("\nMissing values:")
    print(dataset.isna().sum())
    print("\nFeature columns:")
    print(", ".join(dataset.columns))

    return dataset


if __name__ == "__main__":
 build_features()