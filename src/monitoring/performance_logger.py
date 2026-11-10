
import sqlite3
from datetime import datetime, timezone

from src.monitoring.prediction_logger import DB_PATH


def log_request_metric(method, path, status_code, response_time_ms):
    with sqlite3.connect(DB_PATH, timeout=5) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS api_request_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                response_time_ms REAL NOT NULL
            )
        """)

        connection.execute("""
            INSERT INTO api_request_metrics (
                timestamp,
                method,
                path,
                status_code,
                response_time_ms
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            method,
            path,
            status_code,
            response_time_ms
        ))


def log_performance_alert(alert_type, message, endpoint, response_time_ms, status_code):
    with sqlite3.connect(DB_PATH, timeout=5) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS api_performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                response_time_ms REAL NOT NULL,
                status_code INTEGER NOT NULL
            )
        """)

        connection.execute("""
            INSERT INTO api_performance_alerts (
                timestamp,
                alert_type,
                message,
                endpoint,
                response_time_ms,
                status_code
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            alert_type,
            message,
            endpoint,
            response_time_ms,
            status_code
        ))
