import pandas as pd

from src.features import build_features


def test_build_features_creates_correct_order_features(tmp_path, monkeypatch):
    orders = pd.DataFrame({
        "order_id": ["order-1", "order-2", "order-3"],
        "customer_id": ["customer-1", "customer-2", "customer-1"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": [
            "2018-01-01 10:00:00",
            "2018-01-02 12:00:00",
            "2018-01-03 09:00:00",
        ],
        "order_estimated_delivery_date": [
            "2018-01-05", "2018-01-06", "2018-01-07"
        ],
        "order_delivered_customer_date": [
            "2018-01-06", "2018-01-05", None
        ],
    })

    items = pd.DataFrame({
        "order_id": ["order-1", "order-1", "order-2"],
        "order_item_id": [1, 2, 1],
        "seller_id": ["seller-1", "seller-2", "seller-1"],
        "price": [100.0, 50.0, 200.0],
        "freight_value": [10.0, 5.0, 20.0],
    })

    customers = pd.DataFrame({
        "customer_id": ["customer-1", "customer-2"],
        "customer_state": ["SP", "RJ"],
    })

    sellers = pd.DataFrame({
        "seller_id": ["seller-1", "seller-2"],
        "seller_state": ["SP", "MG"],
    })

    datasets = {
        "orders": orders,
        "items": items,
        "customers": customers,
        "sellers": sellers,
    }

    monkeypatch.setattr(
        build_features,
        "load_csv",
        lambda name: datasets[name].copy(),
    )
    monkeypatch.setattr(build_features, "OUTPUT_DIR", tmp_path)

    result = build_features.build_features()

    assert len(result) == 2
    assert set(result["order_id"]) == {"order-1", "order-2"}

    labels = result.set_index("order_id")["is_delayed"]
    assert labels["order-1"] == 1
    assert labels["order-2"] == 0

    delayed_order = result.set_index("order_id").loc["order-1"]
    assert delayed_order["item_count"] == 2
    assert delayed_order["total_price"] == 150.0
    assert delayed_order["total_freight_value"] == 15.0
    assert delayed_order["unique_sellers"] == 2
    assert delayed_order["customer_state"] == "SP"
    assert delayed_order["seller_states"] == 2

    assert result["order_id"].is_unique
    assert "order_delivered_customer_date" not in result.columns
    assert (tmp_path / "delivery_risk_features.csv").exists()
