
from pathlib import Path

import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset
from src.monitoring.drift_alerts import evaluate_drift


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
API_URL = "http://localhost:8000/monitoring/predictions?limit=1000"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "data_drift_report.html"

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
    df = pd.read_csv(REFERENCE_PATH)
    return df[FEATURES].copy()


def load_current_data():
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    records = response.json()["predictions"]

    if not records:
        raise ValueError("No prediction records available.")

    df = pd.DataFrame([
        record["input_features"] for record in records
    ])

    return df[FEATURES].copy()


def generate_drift_report():
    reference_data = load_reference_data()
    current_data = load_current_data()

    if len(current_data) < MINIMUM_RECORDS:
        raise ValueError(
            f"Not enough prediction records for drift analysis. "
            f"Required: {MINIMUM_RECORDS}, "
            f"available: {len(current_data)}"
        )

    print("Reference dataset:", reference_data.shape)
    print("Current dataset:", current_data.shape)

    report = Report(
        metrics=[DataDriftPreset(columns=FEATURES)]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )
    alert_summary = evaluate_drift(result.metric_results)
    print("\nFeature drift scores:")

    for metric_result in result.metric_results.values():
        details = metric_result.dict()

        if details.get("type", "").endswith("SingleValue"):
            print(
                details.get("display_name"),
                "=>",
                details.get("value"),
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))
    print("Drift alert status:", alert_summary["overall_status"])
    print(f"\nDrift report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    generate_drift_report()
