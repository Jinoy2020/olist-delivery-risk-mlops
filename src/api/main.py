
import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "olist-delivery-risk-classifier"
MODEL_VERSION = "1"

# Load the registered model once when the API starts
model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

app = FastAPI(
    title="Olist Delivery Risk API",
    description="Predict whether an Olist order will be delayed.",
    version="1.0.0",
)


# Input schema: must match the training features
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


@app.post("/predict")
def predict(order: OrderFeatures):
    try:
        input_data = pd.DataFrame([order.model_dump()])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        delay_probability = float(probabilities[1])

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
