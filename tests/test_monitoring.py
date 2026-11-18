
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.monitoring.data_drift as data_drift
import src.monitoring.drift_alerts as drift_alerts
import src.monitoring.scheduled_drift as scheduled_drift


def make_metric(display_name, value):
    return SimpleNamespace(
        dict=lambda: {
            "display_name": display_name,
            "value": value,
        }
    )


def test_evaluate_drift_detects_scores_above_threshold(
    tmp_path, monkeypatch
):
    alert_path = tmp_path / "drift_alerts.json"
    monkeypatch.setattr(drift_alerts, "ALERT_PATH", alert_path)

    metrics = {
        "metric_1": make_metric("Value drift for total_price", 0.15),
        "metric_2": make_metric(
            "Value drift for estimated_lead_days", 0.05
        ),
    }

    result = drift_alerts.evaluate_drift(
        metrics,
        current_records=150,
    )

    assert result["overall_status"] == "DRIFT_DETECTED"
    assert result["features_checked"] == 2
    assert result["features_with_drift"] == 1
    assert result["alerts"][0]["feature"] == "total_price"
    assert result["alerts"][0]["drift_score"] == 0.15
    assert alert_path.exists()


def test_drift_score_equal_to_threshold_does_not_trigger_alert(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        drift_alerts,
        "ALERT_PATH",
        tmp_path / "drift_alerts.json",
    )

    metrics = {
        "metric_1": make_metric("Value drift for total_price", 0.10),
    }

    result = drift_alerts.evaluate_drift(metrics)

    assert result["overall_status"] == "NO_DRIFT"
    assert result["features_with_drift"] == 0
    assert result["alerts"] == []


def test_save_monitoring_status_preserves_last_successful_analysis(
    tmp_path, monkeypatch
):
    alert_path = tmp_path / "drift_alerts.json"
    monkeypatch.setattr(drift_alerts, "ALERT_PATH", alert_path)

    previous_analysis = {
        "overall_status": "DRIFT_DETECTED",
        "features_checked": 8,
        "features_with_drift": 2,
        "feature_scores": {"total_price": 0.15},
    }

    alert_path.write_text(
        json.dumps({
            "overall_status": "DRIFT_DETECTED",
            "feature_scores": previous_analysis["feature_scores"],
            "last_successful_analysis": previous_analysis,
        }),
        encoding="utf-8",
    )

    result = drift_alerts.save_monitoring_status(
        status="INSUFFICIENT_DATA",
        current_records=25,
        message="Waiting for more production records.",
    )

    assert result["overall_status"] == "INSUFFICIENT_DATA"
    assert result["current_records"] == 25
    assert result["last_successful_analysis"] == previous_analysis


def test_evaluate_drift_with_no_metrics_preserves_last_successful_analysis(
    tmp_path, monkeypatch
):
    alert_path = tmp_path / "drift_alerts.json"
    monkeypatch.setattr(drift_alerts, "ALERT_PATH", alert_path)

    previous_analysis = {
        "overall_status": "NO_DRIFT",
        "features_checked": 8,
        "features_with_drift": 0,
        "feature_scores": {"total_price": 0.05},
    }

    alert_path.write_text(
        json.dumps({
            "overall_status": "NO_DRIFT",
            "feature_scores": previous_analysis["feature_scores"],
            "last_successful_analysis": previous_analysis,
        }),
        encoding="utf-8",
    )

    result = drift_alerts.evaluate_drift(
        {},
        current_records=120,
    )

    assert result["overall_status"] == "INSUFFICIENT_METRICS"
    assert result["last_successful_analysis"] == previous_analysis


def test_generate_drift_report_skips_when_records_below_minimum(
    monkeypatch,
):
    current_data = Mock()
    current_data.__len__ = Mock(return_value=99)

    monkeypatch.setattr(
        data_drift,
        "load_current_data",
        lambda: current_data,
    )

    reference_loader = Mock()
    monkeypatch.setattr(
        data_drift,
        "load_reference_data",
        reference_loader,
    )

    status_saver = Mock(
        return_value={"overall_status": "INSUFFICIENT_DATA"}
    )
    monkeypatch.setattr(
        data_drift,
        "save_monitoring_status",
        status_saver,
    )

    result = data_drift.generate_drift_report()

    assert result["overall_status"] == "INSUFFICIENT_DATA"
    reference_loader.assert_not_called()
    status_saver.assert_called_once()

    assert status_saver.call_args.kwargs["current_records"] == 99


def test_scheduled_monitoring_handles_errors(caplog, monkeypatch):
    def raise_error():
        raise RuntimeError("Simulated monitoring failure")

    monkeypatch.setattr(
        scheduled_drift,
        "generate_drift_report",
        raise_error,
    )

    with caplog.at_level("ERROR"):
        scheduled_drift.run_monitoring()

    assert "Data drift check failed." in caplog.text
    assert "Simulated monitoring failure" in caplog.text
