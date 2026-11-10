
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app
from fastapi import Request
from collections import defaultdict
from threading import Lock
import sqlite3

import src.monitoring.performance_logger as performance_logger

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
def test_monitoring_metrics():
    response = client.get("/monitoring/metrics")

    assert response.status_code == 200

    result = response.json()

    assert "total_requests" in result
    assert "successful_requests" in result
    assert "error_requests" in result
    assert "error_rate" in result
    assert "average_response_time_ms" in result
    assert "min_response_time_ms" in result
    assert "max_response_time_ms" in result

    assert result["total_requests"] >= 1
    assert result["successful_requests"] >= 1
def test_monitoring_metrics_history():
    response = client.get("/monitoring/metrics/history")

    assert response.status_code == 200

    result = response.json()

    assert "total_returned" in result
    assert "metrics" in result
    assert isinstance(result["metrics"], list)

    if result["metrics"]:
        record = result["metrics"][0]

        assert "timestamp" in record
        assert "method" in record
        assert "path" in record
        assert "status_code" in record
        assert "response_time_ms" in record
def test_performance_alert_persistence(tmp_path, monkeypatch):
    test_db = tmp_path / "test_monitoring.db"

    monkeypatch.setattr(performance_logger, "DB_PATH", test_db)

    performance_logger.log_performance_alert(
        alert_type="HIGH_LATENCY",
        message="API response time exceeded 500 ms",
        endpoint="/predict",
        response_time_ms=650.0,
        status_code=200,
    )

    with sqlite3.connect(test_db) as connection:
        record = connection.execute(
            """
            SELECT alert_type, message, endpoint, response_time_ms, status_code
            FROM api_performance_alerts
            """
        ).fetchone()

    assert record is not None
    assert record[0] == "HIGH_LATENCY"
    assert record[1] == "API response time exceeded 500 ms"
    assert record[2] == "/predict"
    assert record[3] == 650.0
    assert record[4] == 200
def test_monitoring_alerts():
    response = client.get("/monitoring/alerts")

    assert response.status_code == 200

    result = response.json()

    assert "total_returned" in result
    assert "alerts" in result
    assert isinstance(result["alerts"], list)

@patch("src.api.main.get_features_from_feast")
def test_predict_with_feast(mock_get_features):
    mock_get_features.return_value = {
        "purchase_month": 3,
        "purchase_day_of_week": 2,
        "purchase_hour": 14,
        "estimated_lead_days": 20.0,
        "item_count": 1,
        "total_price": 100.0,
        "total_freight_value": 20.0,
        "customer_state": "SP",
    }

    response = client.post(
        "/predict-with-feast/test-order-001"
    )

    assert response.status_code == 200

    result = response.json()

    assert result["order_id"] == "test-order-001"
    assert result["features_source"] == "Feast"
    assert result["is_delayed"] in [0, 1]
    assert 0.0 <= result["delay_probability"] <= 1.0
    assert result["model_name"] == "olist-delivery-risk-classifier"
    assert result["model_version"] == "1"

    mock_get_features.assert_called_once_with("test-order-001")

@patch("src.api.main.get_features_from_feast")
def test_predict_with_feast_missing_features(mock_get_features):
    from fastapi import HTTPException

    mock_get_features.side_effect = HTTPException(
        status_code=404,
        detail="Feature missing",
    )

    response = client.post(
        "/predict-with-feast/missing-order-001"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Feature missing"

@patch("src.api.main.get_features_from_feast")
def test_predict_with_feast_server_unavailable(mock_get_features):
    from fastapi import HTTPException

    mock_get_features.side_effect = HTTPException(
        status_code=503,
        detail="Unable to connect to Feast feature server.",
    )

    response = client.post(
        "/predict-with-feast/test-order-002"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Unable to connect to Feast feature server."
    )