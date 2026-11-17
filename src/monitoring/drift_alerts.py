
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALERT_PATH = PROJECT_ROOT / "reports" / "drift_alerts.json"

DRIFT_THRESHOLD = 0.10


def save_monitoring_status(
    status,
    current_records=0,
    message=None,
):
    """
    Save the latest monitoring status without deleting
    historical successful drift analysis.
    """

    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)

    previous_data = {}

    if ALERT_PATH.exists():
        try:
            previous_data = json.loads(
                ALERT_PATH.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            previous_data = {}

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": status,
        "current_records": current_records,
        "minimum_records": 100,
        "message": message,
        "threshold": DRIFT_THRESHOLD,
        "last_successful_analysis": previous_data.get(
            "last_successful_analysis",
            previous_data if previous_data.get("feature_scores") else None,
        ),
    }

    ALERT_PATH.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8",
    )

    print("Monitoring status:", status)
    print("Current production records:", current_records)

    return result


def evaluate_drift(metric_results, current_records=None):
    alerts = []
    feature_scores = {}

    for metric_result in metric_results.values():
        details = metric_result.dict()

        display_name = details.get("display_name", "")

        if not display_name.startswith("Value drift for "):
            continue

        feature_name = display_name.replace("Value drift for ", "")
        score = float(details["value"])

        feature_scores[feature_name] = round(score, 6)

        if score > DRIFT_THRESHOLD:
            alerts.append({
                "feature": feature_name,
                "drift_score": round(score, 6),
                "threshold": DRIFT_THRESHOLD,
                "status": "DRIFT_DETECTED",
            })

    previous_data = {}

    if ALERT_PATH.exists():
        try:
            previous_data = json.loads(
                ALERT_PATH.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            previous_data = {}

    previous_analysis = previous_data.get(
        "last_successful_analysis"
    )

    if previous_analysis is None and previous_data.get("feature_scores"):
        previous_analysis = previous_data

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": (
            "INSUFFICIENT_METRICS"
            if not feature_scores
            else "DRIFT_DETECTED"
            if alerts
            else "NO_DRIFT"
        ),
        "current_records": current_records,
        "minimum_records": 100,
        "threshold": DRIFT_THRESHOLD,
        "features_checked": len(feature_scores),
        "features_with_drift": len(alerts),
        "feature_scores": feature_scores,
        "alerts": alerts,
        "last_successful_analysis": None,
    }

    result["last_successful_analysis"] = {
        key: value
        for key, value in result.items()
        if key != "last_successful_analysis"
    }

    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ALERT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nDrift alert summary:")
    print("Features checked:", len(feature_scores))
    print("Features with drift:", len(alerts))
    print("Overall status:", result["overall_status"])

    for alert in alerts:
        print(
            f"WARNING: {alert['feature']} "
            f"drift score {alert['drift_score']:.4f} "
            f"exceeds {DRIFT_THRESHOLD}"
        )

    print(f"Alert results saved to: {ALERT_PATH}")

    return result
