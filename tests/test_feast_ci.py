
from datetime import timedelta

import pandas as pd
import pytest

from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


def test_feast_historical_feature_retrieval(tmp_path):
    # Create synthetic feature data
    source_path = tmp_path / "features.parquet"

    source_df = pd.DataFrame({
        "order_id": ["ci-order-001"],
        "event_timestamp": pd.to_datetime(
            ["2026-01-01 10:00:00"], utc=True
        ),
        "purchase_month": [1],
        "item_count": [2],
        "total_price": [150.0],
        "customer_state": ["SP"],
    })

    source_df.to_parquet(source_path, index=False)

    # Create an isolated Feast repository
    repo_path = tmp_path / "feature_repo"
    repo_path.mkdir()

    data_path = repo_path / "data"
    data_path.mkdir()

    (repo_path / "feature_store.yaml").write_text(
        """
project: feast_ci
provider: local
registry: data/registry.db

offline_store:
  type: file

online_store:
  type: sqlite
  path: data/online_store.db
""",
        encoding="utf-8",
    )

    order = Entity(
        name="order",
        join_keys=["order_id"],
        value_type=ValueType.STRING,
    )

    source = FileSource(
        name="ci_order_source",
        path=str(source_path),
        timestamp_field="event_timestamp",
    )

    features = FeatureView(
        name="olist_order_features",
        entities=[order],
        ttl=timedelta(days=3650),
        schema=[
            Field(name="purchase_month", dtype=Int64),
            Field(name="item_count", dtype=Int64),
            Field(name="total_price", dtype=Float32),
            Field(name="customer_state", dtype=String),
        ],
        source=source,
        online=True,
    )

    store = FeatureStore(repo_path=str(repo_path))
    store.apply([order, features])

    # Retrieve historical features
    entity_df = pd.DataFrame({
        "order_id": ["ci-order-001"],
        "event_timestamp": pd.to_datetime(
            ["2026-01-01 10:00:00"], utc=True
        ),
    })

    result = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "olist_order_features:purchase_month",
            "olist_order_features:item_count",
            "olist_order_features:total_price",
            "olist_order_features:customer_state",
        ],
    ).to_df()

    assert len(result) == 1
    assert result.iloc[0]["purchase_month"] == 1
    assert result.iloc[0]["item_count"] == 2
    assert result.iloc[0]["total_price"] == pytest.approx(150.0)
    assert result.iloc[0]["customer_state"] == "SP"
