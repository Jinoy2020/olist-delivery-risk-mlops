
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALERT_PATH = PROJECT_ROOT / "reports" / "drift_alerts.json"

DRIFT_THRESHOLD = 0.10


def evaluate_drift(metric_results):
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

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": DRIFT_THRESHOLD,
        "features_checked": len(feature_scores),
        "features_with_drift": len(alerts),
        "feature_scores": feature_scores,
        "alerts": alerts,
        "overall_status": (
            "DRIFT_DETECTED" if alerts else "NO_DRIFT"
        ),
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
