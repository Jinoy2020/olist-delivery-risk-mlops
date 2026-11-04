
import os
import sqlite3
from fastapi import Query
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.monitoring.prediction_logger import DB_PATH, log_prediction

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
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }
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
