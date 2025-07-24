# src/api/main.py

from fastapi import FastAPI
from src.api.pydantic_models import TransactionFeatures, PredictionResponse
import mlflow.pyfunc
import pandas as pd


mlflow.set_tracking_uri("http://host.docker.internal:5000")
# mlflow.set_tracking_uri("file:/app/mlruns")

app = FastAPI()

# Load the model from the MLflow registry
model_name = "FraudDetectionModel"
model_stage = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.get("/")
def read_root():
    return {"message": "Fraud risk prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = model.predict(df)
    return PredictionResponse(risk_probability=float(prediction[0]))
