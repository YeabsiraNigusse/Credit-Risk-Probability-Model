import os
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException

from pydantic_models import TransactionFeatures, PredictionResponse
from mlflow.exceptions import MlflowException

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI()
model_name  = "credit_fraud_model"
model_stage = "Production"

try:
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
    print(f"✅ Loaded {model_name}@{model_stage}")
except MlflowException as e:
    print(f"⚠️  Could not load model: {e}")
    model = None

@app.get("/")
async def health():
    return {"message": "API running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: TransactionFeatures):
    if model is None:
        raise HTTPException(503, "Model not available")

    # Build DF from the raw JSON
    df = pd.DataFrame([features.dict()])

    try:
        proba = float(model.predict(df)[0])
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    return PredictionResponse(risk_probability=proba)
