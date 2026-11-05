
import os
import sqlite3
from fastapi import Query
import mlflow
import mlflow.sklearn
import pandas as pd
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.monitoring.prediction_logger import DB_PATH, log_prediction
from threading import Lock
from fastapi import Request
from src.monitoring.performance_logger import (
    log_request_metric,
    log_performance_alert,
)

MODEL_NAME = "olist-delivery-risk-classifier"
MODEL_VERSION = "1"

# Load the model from a local artifact directory in Docker.
# Otherwise, load the registered model from local MLflow.
MODEL_PATH = os.getenv("MODEL_PATH")

if MODEL_PATH:
    model = mlflow.sklearn.load_model(MODEL_PATH)
else:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )


app = FastAPI(
    title="Olist Delivery Risk API",
    description="Predict whether an Olist order will be delayed.",
    version="1.0.0",
)

# --------------------------------------------------
# API Performance Monitoring
# --------------------------------------------------

request_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "error_requests": 0,
    "total_response_time_ms": 0.0,
    "min_response_time_ms": None,
    "max_response_time_ms": 0.0,
}

metrics_lock = Lock()



@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        response_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        with metrics_lock:
            request_metrics["total_requests"] += 1
            request_metrics["error_requests"] += 1
            request_metrics["total_response_time_ms"] += response_time_ms

        log_request_metric(
            request.method,
            request.url.path,
            500,
            response_time_ms
        )
        log_performance_alert(
            "HTTP_ERROR",
            "Unhandled API exception detected",
            request.url.path,
            response_time_ms,
            500
        )

        raise

    response_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    with metrics_lock:
        request_metrics["total_requests"] += 1
        request_metrics["total_response_time_ms"] += response_time_ms

        if response.status_code < 400:
            request_metrics["successful_requests"] += 1
        else:
            request_metrics["error_requests"] += 1

        current_min = request_metrics["min_response_time_ms"]

        if current_min is None or response_time_ms < current_min:
            request_metrics["min_response_time_ms"] = response_time_ms

        request_metrics["max_response_time_ms"] = max(
            request_metrics["max_response_time_ms"],
            response_time_ms,
        )

    log_request_metric(
        request.method,
        request.url.path,
        response.status_code,
        response_time_ms
    )
    if response.status_code >= 400:
        log_performance_alert(
            "HTTP_ERROR",
            f"HTTP {response.status_code} error detected",
            request.url.path,
            response_time_ms,
            response.status_code
        )

    if response_time_ms > 500:
        log_performance_alert(
            "HIGH_LATENCY",
            "API response time exceeded 500 ms",
            request.url.path,
            response_time_ms,
            response.status_code
        )

    response.headers["X-Response-Time-MS"] = str(
        round(response_time_ms, 2)
    )

    return response


class OrderFeatures(BaseModel):
    purchase_month: int = Field(ge=1, le=12)
    purchase_day_of_week: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)
    estimated_lead_days: float = Field(gt=0)
    item_count: int = Field(ge=1)
    total_price: float = Field(ge=0)
    total_freight_value: float = Field(ge=0)
    customer_state: str = Field(min_length=2, max_length=2)


@app.get("/")
def root():
    return {
        "message": "Olist Delivery Risk API is running",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.get("/health")
def health():
    start_time = time.perf_counter()

    model_loaded = model is not None
    database_connected = False

    try:
        with sqlite3.connect(DB_PATH, timeout=3) as connection:
            connection.execute("SELECT 1")
            database_connected = True
    except sqlite3.Error:
        database_connected = False

    response_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    is_healthy = model_loaded and database_connected

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_loaded,
        "database_connected": database_connected,
        "health_check_response_time_ms": round(response_time_ms, 2),
    }

@app.get("/monitoring/metrics")
def get_monitoring_metrics():
    with metrics_lock:
        total = request_metrics["total_requests"]

        average_response_time = (
            request_metrics["total_response_time_ms"] / total
            if total > 0 else 0.0
        )

        return {
            "total_requests": total,
            "successful_requests": request_metrics["successful_requests"],
            "error_requests": request_metrics["error_requests"],
            "error_rate": (
                request_metrics["error_requests"] / total
                if total > 0 else 0.0
            ),
            "average_response_time_ms": round(
                average_response_time, 2
            ),
            "min_response_time_ms": (
                round(request_metrics["min_response_time_ms"], 2)
                if request_metrics["min_response_time_ms"] is not None
                else None
            ),
            "max_response_time_ms": round(
                request_metrics["max_response_time_ms"], 2
            ),
        }

@app.get("/monitoring/metrics/history")
def get_metrics_history(limit: int = Query(default=500, ge=1, le=5000)):
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT timestamp, method, path, status_code, response_time_ms
                FROM api_request_metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "total_returned": len(rows),
            "metrics": [
                {
                    "timestamp": row["timestamp"],
                    "method": row["method"],
                    "path": row["path"],
                    "status_code": row["status_code"],
                    "response_time_ms": row["response_time_ms"],
                }
                for row in reversed(rows)
            ],
        }

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve API performance history.",
        ) from exc
@app.get("/monitoring/alerts")
def get_performance_alerts(
    limit: int = Query(default=100, ge=1, le=1000)
):
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT id, timestamp, alert_type, message,
                       endpoint, response_time_ms, status_code
                FROM api_performance_alerts
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "total_returned": len(rows),
            "alerts": [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "alert_type": row["alert_type"],
                    "message": row["message"],
                    "endpoint": row["endpoint"],
                    "response_time_ms": row["response_time_ms"],
                    "status_code": row["status_code"],
                }
                for row in rows
            ],
        }

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve API performance alerts.",
        ) from exc

@app.get("/monitoring/predictions")
def get_monitoring_predictions(limit: int = Query(default=100, ge=1, le=1000)):
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT timestamp, input_features, prediction,
                       delay_probability, model_name, model_version
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "total_returned": len(rows),
            "predictions": [
                {
                    "timestamp": row["timestamp"],
                    "input_features": __import__("json").loads(
                        row["input_features"]
                    ),
                    "prediction": row["prediction"],
                    "delay_probability": row["delay_probability"],
                    "model_name": row["model_name"],
                    "model_version": row["model_version"],
                }
                for row in rows
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve monitoring predictions.",
        ) from exc

@app.post("/predict")
def predict(order: OrderFeatures):

    try:
        input_data = pd.DataFrame([order.model_dump()])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        delay_probability = float(probabilities[1])

        log_prediction(
            input_features=order.model_dump(),
            prediction=prediction,
            delay_probability=delay_probability,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )

        return {
            "is_delayed": prediction,
            "prediction": (
                "Delayed" if prediction == 1 else "On Time"
            ),
            "delay_probability": round(delay_probability, 4),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
