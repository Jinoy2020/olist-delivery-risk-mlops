
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
]


def load_csv(filename: str) -> pd.DataFrame:
    """Load a required Olist CSV file."""
    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset: {file_path}")

    return pd.read_csv(file_path)


def main() -> None:
    for filename in REQUIRED_FILES:
        df = load_csv(filename)
        print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
