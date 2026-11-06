from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE_PATH = PROJECT_ROOT / "feature_store"
SOURCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feast_order_features.parquet"
)

FEATURES = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "estimated_lead_days",
    "item_count",
    "total_price",
    "total_freight_value",
    "customer_state",
]


def test_historical_features_match_source():

    feast = pytest.importorskip("feast")

    if not SOURCE_PATH.exists():
        pytest.skip("Feast Parquet source is not available.")

    registry_path = FEATURE_STORE_PATH / "data" / "registry.db"

    if not registry_path.exists():
        pytest.skip("Feast registry is not available. Run feast apply first.")

    source_df = pd.read_parquet(SOURCE_PATH)
    sample = source_df.iloc[0]

    entity_df = pd.DataFrame({
        "order_id": [sample["order_id"]],
        "event_timestamp": [sample["event_timestamp"]],
    })

    store = feast.FeatureStore(
        repo_path=str(FEATURE_STORE_PATH)
    )

    retrieval_job = store.get_historical_features(
        entity_df=entity_df,
        features=[
            f"olist_order_features:{feature}"
            for feature in FEATURES
        ],
    )

    historical_df = retrieval_job.to_df()

    assert len(historical_df) == 1

    result = historical_df.iloc[0]

    assert result["order_id"] == sample["order_id"]

    for feature in FEATURES:

        if feature in [
            "estimated_lead_days",
            "total_price",
            "total_freight_value",
        ]:
            assert result[feature] == pytest.approx(
                sample[feature], rel=1e-5
            )
        else:
            assert result[feature] == sample[feature]