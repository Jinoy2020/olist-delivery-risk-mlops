
import json
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(
    os.getenv("MONITORING_DB_PATH", "monitoring.db")
)


def log_prediction(
    input_features: dict,
    prediction: int,
    delay_probability: float,
    model_name: str,
    model_version: str,
    order_id: str | None = None,
):
    with sqlite3.connect(DB_PATH, timeout=5) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                order_id TEXT,
                input_features TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                delay_probability REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
        """)

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(predictions)"
            ).fetchall()
        }

        if "order_id" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN order_id TEXT"
            )

        connection.execute("""
            INSERT INTO predictions (
                timestamp,
                order_id,
                input_features,
                prediction,
                delay_probability,
                model_name,
                model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            order_id,
            json.dumps(input_features),
            prediction,
            delay_probability,
            model_name,
            model_version,
        ))
