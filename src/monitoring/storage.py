import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORING_DATA_DIR = Path(
    os.getenv(
        "MONITORING_DATA_DIR",
        str(PROJECT_ROOT / "reports"),
    )
)

DRIFT_ALERT_PATH = MONITORING_DATA_DIR / "drift_alerts.json"
DRIFT_REPORT_PATH = MONITORING_DATA_DIR / "data_drift_report.html"