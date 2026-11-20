
import os
from pathlib import Path

import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset

from src.monitoring.storage import DRIFT_REPORT_PATH
from src.monitoring.drift_alerts import (
    evaluate_drift,
    save_monitoring_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = Path(
    os.getenv(
        "DRIFT_REFERENCE_PATH",
        str(PROJECT_ROOT / "data" / "processed" / "train.csv"),
    )
)

API_BASE_URL = os.getenv(
    "DRIFT_API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

API_URL = (
    f"{API_BASE_URL}/monitoring/predictions"
    "?limit=1000&traffic_source=production"
)

REPORT_PATH = DRIFT_REPORT_PATH

FEATURES = [
    "purchase_month",
    "purchase_day_of_week",
    "purchase_hour",
    "estimated_lead_days",
    "item_count",
    "total_price",
    "total_freight_value",
    "customer_state",
]

MINIMUM_RECORDS = 100


def load_reference_data():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_PATH}"
        )

    df = pd.read_csv(REFERENCE_PATH)
    return df[FEATURES].copy()


def load_current_data():
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    records = response.json()["predictions"]

    if not records:
        return pd.DataFrame(columns=FEATURES)

    df = pd.DataFrame([
        record["input_features"] for record in records
    ])

    return df[FEATURES].copy()


def generate_drift_report():
    current_data = load_current_data()

    if len(current_data) < MINIMUM_RECORDS:
        status = save_monitoring_status(
            status="INSUFFICIENT_DATA",
            current_records=len(current_data),
            message=(
                f"Waiting for at least {MINIMUM_RECORDS} "
                "production prediction records."
            ),
        )

        print("\nDrift analysis skipped.")

        message = status.get(
            "message",
            f"Waiting for at least {MINIMUM_RECORDS} "
            "production prediction records."
        )

        print(message)
        return status

    reference_data = load_reference_data()

    print("Reference dataset:", reference_data.shape)
    print("Current dataset:", current_data.shape)

    report = Report(
        metrics=[DataDriftPreset(columns=FEATURES)]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    alert_summary = evaluate_drift(
        result.metric_results,
        current_records=len(current_data),
    )

    print("\nFeature drift scores:")

    for metric_result in result.metric_results.values():
        details = metric_result.dict()

        if details.get("type", "").endswith("SingleValue"):
            print(
                details.get("display_name"),
                "=>",
                details.get("value"),
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))

    print("Drift alert status:", alert_summary["overall_status"])
    print(f"\nDrift report saved to: {REPORT_PATH}")

    return alert_summary


if __name__ == "__main__":
    generate_drift_report()
