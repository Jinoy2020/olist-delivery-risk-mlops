
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String
from feast import FileSource
from pathlib import Path
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


# Model input features
order_features = FeatureView(
    name="olist_order_features",
    entities=[order],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="purchase_month", dtype=Int64),
        Field(name="purchase_day_of_week", dtype=Int64),
        Field(name="purchase_hour", dtype=Int64),
        Field(name="estimated_lead_days", dtype=Float32),
        Field(name="item_count", dtype=Int64),
        Field(name="total_price", dtype=Float32),
        Field(name="total_freight_value", dtype=Float32),
        Field(name="customer_state", dtype=String),
    ],
    source=order_source,
    online=True,
    description="Checkout-time features for Olist delivery risk prediction",
)
