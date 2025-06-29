"""
Pydantic Models for Credit Risk Assessment API

This module defines the data models used for API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk category enumeration"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class ModelName(str, Enum):
    """Available model names"""
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class CustomerData(BaseModel):
    """
    Customer data model for credit risk assessment
    """
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    
    # RFM Metrics
    recency: float = Field(..., ge=0, description="Days since last transaction")
    frequency: int = Field(..., ge=1, description="Number of transactions")
    monetary_total: float = Field(..., ge=0, description="Total monetary value")
    monetary_avg: float = Field(..., ge=0, description="Average transaction amount")
    
    # RFM Scores
    r_score: int = Field(..., ge=1, le=5, description="Recency score (1-5)")
    f_score: int = Field(..., ge=1, le=5, description="Frequency score (1-5)")
    m_score: int = Field(..., ge=1, le=5, description="Monetary score (1-5)")
    
    # Transaction Features
    total_transactions: int = Field(..., ge=1, description="Total number of transactions")
    total_amount: float = Field(..., ge=0, description="Total transaction amount")
    avg_amount: float = Field(..., ge=0, description="Average transaction amount")
    std_amount: float = Field(..., ge=0, description="Standard deviation of amounts")
    min_amount: float = Field(..., ge=0, description="Minimum transaction amount")
    max_amount: float = Field(..., ge=0, description="Maximum transaction amount")
    
    # Customer Behavior Features
    customer_tenure_days: int = Field(..., ge=0, description="Customer tenure in days")
    transaction_velocity: float = Field(..., ge=0, description="Transactions per day")
    avg_transaction_hour: float = Field(..., ge=0, le=23, description="Average transaction hour")
    weekend_transaction_ratio: float = Field(..., ge=0, le=1, description="Ratio of weekend transactions")
    
    # Derived Features
    rfm_combined_score: int = Field(..., ge=3, le=15, description="Combined RFM score")
    amount_consistency: float = Field(..., ge=0, description="Amount consistency metric")
    
    @validator('monetary_avg')
    def validate_monetary_avg(cls, v, values):
        """Validate that monetary_avg is consistent with total and frequency"""
        if 'monetary_total' in values and 'frequency' in values:
            expected_avg = values['monetary_total'] / values['frequency']
            if abs(v - expected_avg) > 0.01:  # Allow small floating point differences
                raise ValueError('monetary_avg must equal monetary_total / frequency')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "recency": 15,
                "frequency": 25,
                "monetary_total": 2500.0,
                "monetary_avg": 100.0,
                "r_score": 4,
                "f_score": 4,
                "m_score": 3,
                "total_transactions": 25,
                "total_amount": 2500.0,
                "avg_amount": 100.0,
                "std_amount": 25.5,
                "min_amount": 50.0,
                "max_amount": 200.0,
                "customer_tenure_days": 180,
                "transaction_velocity": 0.14,
                "avg_transaction_hour": 14.5,
                "weekend_transaction_ratio": 0.2,
                "rfm_combined_score": 11,
                "amount_consistency": 0.255
            }
        }


class AssessmentRequest(BaseModel):
    """
    Credit risk assessment request model
    """
    customer_data: CustomerData = Field(..., description="Customer data for assessment")
    model_name: ModelName = Field(ModelName.XGBOOST, description="Model to use for assessment")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_data": {
                    "customer_id": "CUST_001",
                    "recency": 15,
                    "frequency": 25,
                    "monetary_total": 2500.0,
                    "monetary_avg": 100.0,
                    "r_score": 4,
                    "f_score": 4,
                    "m_score": 3,
                    "total_transactions": 25,
                    "total_amount": 2500.0,
                    "avg_amount": 100.0,
                    "std_amount": 25.5,
                    "min_amount": 50.0,
                    "max_amount": 200.0,
                    "customer_tenure_days": 180,
                    "transaction_velocity": 0.14,
                    "avg_transaction_hour": 14.5,
                    "weekend_transaction_ratio": 0.2,
                    "rfm_combined_score": 11,
                    "amount_consistency": 0.255
                },
                "model_name": "xgboost"
            }
        }


class AssessmentResponse(BaseModel):
    """
    Credit risk assessment response model
    """
    customer_id: str = Field(..., description="Customer identifier")
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of default")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    risk_category: RiskCategory = Field(..., description="Risk category")
    recommended_loan_amount: int = Field(..., ge=0, description="Recommended loan amount")
    recommended_loan_duration: int = Field(..., ge=1, description="Recommended loan duration in days")
    loan_approved: bool = Field(..., description="Loan approval recommendation")
    assessment_date: datetime = Field(..., description="Assessment timestamp")
    model_used: ModelName = Field(..., description="Model used for assessment")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "risk_probability": 0.15,
                "credit_score": 720,
                "risk_category": "Low",
                "recommended_loan_amount": 5000,
                "recommended_loan_duration": 180,
                "loan_approved": True,
                "assessment_date": "2024-01-15T10:30:00",
                "model_used": "xgboost"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response model
    """
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    available_models: List[str] = Field(..., description="List of available models")


class ModelMetrics(BaseModel):
    """
    Model performance metrics
    """
    model_name: str = Field(..., description="Model name")
    auc: float = Field(..., ge=0, le=1, description="Area Under Curve")
    f1_score: float = Field(..., ge=0, le=1, description="F1 Score")
    accuracy: float = Field(..., ge=0, le=1, description="Accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision")
    recall: float = Field(..., ge=0, le=1, description="Recall")


class FeatureImportance(BaseModel):
    """
    Feature importance model
    """
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")


class PredictionExplanation(BaseModel):
    """
    Prediction explanation model
    """
    customer_id: str = Field(..., description="Customer identifier")
    model_used: str = Field(..., description="Model used for prediction")
    risk_probability: float = Field(..., ge=0, le=1, description="Predicted risk probability")
    credit_score: int = Field(..., ge=300, le=850, description="Predicted credit score")
    top_features: List[FeatureImportance] = Field(..., description="Top contributing features")
    explanation: str = Field(..., description="Human-readable explanation")


class ErrorResponse(BaseModel):
    """
    Error response model
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data provided",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
