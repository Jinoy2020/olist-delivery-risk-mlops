
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType


# Olist order entity
order = Entity(
    name="order",
    join_keys=["order_id"],
    value_type=ValueType.STRING,
    description="Unique Olist order identifier",
)


# Historical feature data
project_root = Path(__file__).resolve().parents[2]

order_source = FileSource(
    name="olist_order_features_source",
    path=str(
        project_root / "data" / "processed" / "feast_order_features.parquet"
    ),
    timestamp_field="event_timestamp",
)


# Shared model input feature schema
order_feature_schema = [
    Field(name="order_id", dtype=String),
    Field(name="purchase_month", dtype=Int64),
    Field(name="purchase_day_of_week", dtype=Int64),
    Field(name="purchase_hour", dtype=Int64),
    Field(name="estimated_lead_days", dtype=Float32),
    Field(name="item_count", dtype=Int64),
    Field(name="total_price", dtype=Float32),
    Field(name="total_freight_value", dtype=Float32),
    Field(name="customer_state", dtype=String),
]


# Historical feature view - unchanged name and source
order_features = FeatureView(
    name="olist_order_features",
    entities=[order],
    ttl=timedelta(days=3650),
    schema=order_feature_schema,
    source=order_source,
    online=True,
    description="Historical checkout features for Olist delivery risk prediction",
)


# Live push source backed by historical Parquet data
order_push_source = PushSource(
    name="olist_order_push_source",
    batch_source=order_source,
    description="Live checkout feature ingestion for Olist orders",
)


# Live feature view
order_live_features = FeatureView(
    name="olist_order_live_features",
    entities=[order],
    ttl=timedelta(days=3650),
    schema=order_feature_schema,
    source=order_push_source,
    online=True,
    description="Live checkout features for Olist delivery risk prediction",
)
