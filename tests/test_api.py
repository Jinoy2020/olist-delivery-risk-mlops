
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Olist Delivery Risk API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict():
    payload = {
        "purchase_month": 3,
        "purchase_day_of_week": 2,
        "purchase_hour": 14,
        "estimated_lead_days": 20.0,
        "item_count": 1,
        "total_price": 100.0,
        "total_freight_value": 20.0,
        "customer_state": "SP",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["is_delayed"] in [0, 1]
    assert result["prediction"] in ["Delayed", "On Time"]
    assert 0.0 <= result["delay_probability"] <= 1.0
    assert result["model_name"] == "olist-delivery-risk-classifier"
    assert result["model_version"] == "1"


def test_predict_invalid_input():
    payload = {
        "purchase_month": 15,
        "purchase_day_of_week": 2,
        "purchase_hour": 14,
        "estimated_lead_days": 20.0,
        "item_count": 1,
        "total_price": 100.0,
        "total_freight_value": 20.0,
        "customer_state": "SP",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
