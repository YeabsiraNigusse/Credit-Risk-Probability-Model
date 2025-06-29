"""
FastAPI Application for Credit Risk Assessment

This module provides REST API endpoints for credit risk assessment,
including risk probability calculation and credit scoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import CreditRiskPredictor
from .pydantic_models import (
    CustomerData, AssessmentRequest, AssessmentResponse,
    HealthResponse, ModelMetrics, PredictionExplanation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="API for credit risk assessment and loan recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[CreditRiskPredictor] = None


def get_predictor() -> CreditRiskPredictor:
    """
    Dependency to get the predictor instance
    """
    global predictor
    if predictor is None:
        try:
            predictor = CreditRiskPredictor()
            logger.info("Predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return predictor


@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup
    """
    logger.info("Starting Credit Risk Assessment API")
    try:
        global predictor
        predictor = CreditRiskPredictor()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if models are loaded
        global predictor
        models_loaded = predictor is not None and len(predictor.models) > 0
        
        return HealthResponse(
            status="healthy" if models_loaded else "unhealthy",
            timestamp=datetime.now(),
            models_loaded=models_loaded,
            available_models=list(predictor.models.keys()) if predictor else []
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            models_loaded=False,
            available_models=[]
        )


@app.post("/assess", response_model=AssessmentResponse)
async def assess_credit_risk(
    request: AssessmentRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Assess credit risk for a customer
    """
    try:
        logger.info(f"Processing assessment request for customer: {request.customer_data.customer_id}")
        
        # Convert customer data to DataFrame
        customer_dict = request.customer_data.dict()
        customer_df = pd.DataFrame([customer_dict])
        
        # Remove customer_id from features if present
        if 'customer_id' in customer_df.columns:
            customer_id = customer_df['customer_id'].iloc[0]
            feature_df = customer_df.drop('customer_id', axis=1)
        else:
            customer_id = "unknown"
            feature_df = customer_df
        
        # Perform assessment
        results = predictor.comprehensive_assessment(
            feature_df, 
            model_name=request.model_name
        )
        
        # Extract results
        result = results.iloc[0]
        
        response = AssessmentResponse(
            customer_id=customer_id,
            risk_probability=float(result['risk_probability']),
            credit_score=int(result['credit_score']),
            risk_category=result['risk_category'],
            recommended_loan_amount=int(result['recommended_loan_amount']),
            recommended_loan_duration=int(result['recommended_loan_duration']),
            loan_approved=bool(result['loan_approved']),
            assessment_date=result['assessment_date'],
            model_used=request.model_name
        )
        
        logger.info(f"Assessment completed for customer {customer_id}")
        return response
        
    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@app.post("/batch_assess", response_model=List[AssessmentResponse])
async def batch_assess_credit_risk(
    requests: List[AssessmentRequest],
    background_tasks: BackgroundTasks,
    predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Batch assess credit risk for multiple customers
    """
    try:
        logger.info(f"Processing batch assessment for {len(requests)} customers")
        
        responses = []
        
        for request in requests:
            # Convert customer data to DataFrame
            customer_dict = request.customer_data.dict()
            customer_df = pd.DataFrame([customer_dict])
            
            # Remove customer_id from features if present
            if 'customer_id' in customer_df.columns:
                customer_id = customer_df['customer_id'].iloc[0]
                feature_df = customer_df.drop('customer_id', axis=1)
            else:
                customer_id = "unknown"
                feature_df = customer_df
            
            # Perform assessment
            results = predictor.comprehensive_assessment(
                feature_df, 
                model_name=request.model_name
            )
            
            # Extract results
            result = results.iloc[0]
            
            response = AssessmentResponse(
                customer_id=customer_id,
                risk_probability=float(result['risk_probability']),
                credit_score=int(result['credit_score']),
                risk_category=result['risk_category'],
                recommended_loan_amount=int(result['recommended_loan_amount']),
                recommended_loan_duration=int(result['recommended_loan_duration']),
                loan_approved=bool(result['loan_approved']),
                assessment_date=result['assessment_date'],
                model_used=request.model_name
            )
            
            responses.append(response)
        
        logger.info(f"Batch assessment completed for {len(responses)} customers")
        return responses
        
    except Exception as e:
        logger.error(f"Batch assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch assessment failed: {str(e)}")


@app.get("/explain/{customer_id}", response_model=PredictionExplanation)
async def explain_prediction(
    customer_id: str,
    model_name: str = "xgboost",
    predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Explain prediction for a specific customer
    """
    try:
        # This would require storing customer data or receiving it as input
        # For now, return a placeholder response
        logger.info(f"Explanation requested for customer {customer_id}")
        
        # In a real implementation, you would:
        # 1. Retrieve customer data from database
        # 2. Generate explanation using predictor.explain_prediction()
        # 3. Return formatted explanation
        
        return PredictionExplanation(
            customer_id=customer_id,
            model_used=model_name,
            risk_probability=0.0,
            credit_score=650,
            top_features=[],
            explanation="Explanation feature requires customer data storage"
        )
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/models", response_model=Dict[str, Any])
async def get_available_models(
    predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Get information about available models
    """
    try:
        return {
            "available_models": list(predictor.models.keys()),
            "model_metrics": predictor.model_metrics,
            "default_model": "xgboost"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/metrics/{model_name}", response_model=ModelMetrics)
async def get_model_metrics(
    model_name: str,
    predictor: CreditRiskPredictor = Depends(get_predictor)
):
    """
    Get metrics for a specific model
    """
    try:
        if model_name not in predictor.model_metrics:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        metrics = predictor.model_metrics[model_name]
        
        return ModelMetrics(
            model_name=model_name,
            auc=metrics.get('auc', 0.0),
            f1_score=metrics.get('f1', 0.0),
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model metrics")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
