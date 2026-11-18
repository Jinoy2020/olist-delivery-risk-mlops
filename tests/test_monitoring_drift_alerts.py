from src.monitoring import drift_alerts


class FakeMetricResult:
    def __init__(self, display_name, value):
        self.display_name = display_name
        self.value = value

    def dict(self):
        return {
            "display_name": self.display_name,
            "value": self.value,
        }


def test_empty_metrics_preserve_last_successful_analysis(tmp_path, monkeypatch):
    alert_path = tmp_path / "drift_alerts.json"
    previous_analysis = {
        "timestamp": "2026-11-17T10:00:00+00:00",
        "overall_status": "NO_DRIFT",
        "features_checked": 2,
        "features_with_drift": 0,
        "feature_scores": {"total_price": 0.01, "item_count": 0.02},
        "alerts": [],
    }

    alert_path.write_text(
        __import__("json").dumps({
            "overall_status": "NO_DRIFT",
            "last_successful_analysis": previous_analysis,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift_alerts, "ALERT_PATH", alert_path)

    result = drift_alerts.evaluate_drift({}, current_records=100)

    assert result["overall_status"] == "INSUFFICIENT_METRICS"
    assert result["feature_scores"] == {}
    assert result["last_successful_analysis"] == previous_analysis


def test_valid_metrics_replace_last_successful_analysis(tmp_path, monkeypatch):
    alert_path = tmp_path / "drift_alerts.json"
    previous_analysis = {
        "overall_status": "NO_DRIFT",
        "feature_scores": {"total_price": 0.01},
    }

    alert_path.write_text(
        __import__("json").dumps({
            "last_successful_analysis": previous_analysis,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift_alerts, "ALERT_PATH", alert_path)

    result = drift_alerts.evaluate_drift(
        {
            "metric_1": FakeMetricResult(
                "Value drift for total_price",
                0.25,
            )
        },
        current_records=120,
    )

    assert result["overall_status"] == "DRIFT_DETECTED"
    assert result["last_successful_analysis"]["feature_scores"] == {
        "total_price": 0.25
    }
    assert result["last_successful_analysis"] != previous_analysis
