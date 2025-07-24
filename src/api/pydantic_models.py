# src/api/pydantic_models.py

from pydantic import BaseModel
from typing import List

class TransactionFeatures(BaseModel):
    recency_days: float
    frequency: int
    monetary: float
    fraud_ratio: float

class PredictionResponse(BaseModel):
    risk_probability: float
