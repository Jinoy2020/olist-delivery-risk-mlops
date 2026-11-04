import json
import os
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = Path(
    os.getenv(
        "MONITORING_DB_PATH",
        PROJECT_ROOT / "monitoring_volume.db",
    )
)


def generate_summary():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(
            "SELECT * FROM predictions",
            connection,
        )

    if df.empty:
        raise ValueError("No prediction records found.")

    summary = {
        "total_predictions": int(len(df)),
        "delayed_predictions": int((df["prediction"] == 1).sum()),
        "on_time_predictions": int((df["prediction"] == 0).sum()),
        "average_delay_probability": round(
            float(df["delay_probability"].mean()), 4
        ),
        "model_versions": sorted(
            df["model_version"].unique().tolist()
        ),
        "latest_prediction_timestamp": df["timestamp"].max(),
    }

    output_path = PROJECT_ROOT / "reports" / "monitoring_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(json.dumps(summary, indent=4))
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    generate_summary()