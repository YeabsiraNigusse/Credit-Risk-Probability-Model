"""
Prediction Module for Credit Risk Assessment

This module contains functions for making predictions using trained credit risk models,
including risk probability calculation and credit score assignment.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Credit Risk Prediction and Scoring Class
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the predictor with trained models
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.load_models()
        
    def load_models(self):
        """
        Load trained models and associated artifacts
        """
        logger.info(f"Loading models from {self.model_dir}")
        
        model_files = {
            'logistic_regression': 'logistic_regression_model.joblib',
            'random_forest': 'random_forest_model.joblib',
            'xgboost': 'xgboost_model.joblib'
        }
        
        scaler_files = {
            'logistic_regression': 'logistic_regression_scaler.joblib'
        }
        
        # Load models
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        # Load scalers
        for model_name, filename in scaler_files.items():
            scaler_path = os.path.join(self.model_dir, filename)
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
                logger.info(f"Loaded {model_name} scaler")
        
        # Load metadata
        metrics_path = os.path.join(self.model_dir, 'model_metrics.joblib')
        if os.path.exists(metrics_path):
            self.model_metrics = joblib.load(metrics_path)
        
        importance_path = os.path.join(self.model_dir, 'feature_importance.joblib')
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
    
    def predict_risk_probability(self, X: pd.DataFrame, 
                                model_name: str = 'xgboost') -> np.ndarray:
        """
        Predict risk probability for given features
        
        Args:
            X: Input features
            model_name: Name of the model to use
            
        Returns:
            Array of risk probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Apply scaling if needed
        if model_name in self.scalers:
            X_processed = self.scalers[model_name].transform(X)
        else:
            X_processed = X
        
        # Get probability of default (class 1)
        risk_probabilities = model.predict_proba(X_processed)[:, 1]
        
        return risk_probabilities
    
    def calculate_credit_score(self, risk_probabilities: np.ndarray,
                              score_range: Tuple[int, int] = (300, 850)) -> np.ndarray:
        """
        Convert risk probabilities to credit scores
        
        Args:
            risk_probabilities: Array of risk probabilities (0-1)
            score_range: Tuple of (min_score, max_score)
            
        Returns:
            Array of credit scores
        """
        min_score, max_score = score_range
        
        # Convert probability to score (higher probability = lower score)
        # Using logistic transformation for better distribution
        odds = risk_probabilities / (1 - risk_probabilities + 1e-10)
        log_odds = np.log(odds + 1e-10)
        
        # Normalize to score range
        # Typical credit score distribution: mean around 650-700
        base_score = 650
        scale_factor = 100
        
        credit_scores = base_score - (log_odds * scale_factor)
        
        # Clip to valid range
        credit_scores = np.clip(credit_scores, min_score, max_score)
        
        return credit_scores.astype(int)
    
    def predict_loan_amount(self, X: pd.DataFrame, 
                           risk_probabilities: np.ndarray,
                           base_amount: float = 1000,
                           max_amount: float = 10000) -> np.ndarray:
        """
        Predict optimal loan amount based on risk and customer profile
        
        Args:
            X: Customer features
            risk_probabilities: Risk probabilities
            base_amount: Base loan amount
            max_amount: Maximum loan amount
            
        Returns:
            Array of recommended loan amounts
        """
        # Risk-adjusted loan amounts
        risk_multiplier = 1 - risk_probabilities
        
        # Consider customer monetary value if available
        if 'monetary_total' in X.columns:
            monetary_factor = np.log1p(X['monetary_total']) / 10
            monetary_factor = np.clip(monetary_factor, 0.5, 2.0)
        else:
            monetary_factor = 1.0
        
        # Consider transaction frequency
        if 'frequency' in X.columns:
            frequency_factor = np.log1p(X['frequency']) / 5
            frequency_factor = np.clip(frequency_factor, 0.5, 1.5)
        else:
            frequency_factor = 1.0
        
        # Calculate loan amount
        loan_amounts = base_amount * risk_multiplier * monetary_factor * frequency_factor
        loan_amounts = np.clip(loan_amounts, base_amount * 0.1, max_amount)
        
        return loan_amounts.astype(int)
    
    def predict_loan_duration(self, X: pd.DataFrame,
                             risk_probabilities: np.ndarray,
                             min_duration: int = 30,
                             max_duration: int = 365) -> np.ndarray:
        """
        Predict optimal loan duration based on risk and customer profile
        
        Args:
            X: Customer features
            risk_probabilities: Risk probabilities
            min_duration: Minimum loan duration in days
            max_duration: Maximum loan duration in days
            
        Returns:
            Array of recommended loan durations
        """
        # Lower risk customers get longer durations
        risk_factor = 1 - risk_probabilities
        
        # Consider customer loyalty (recency and frequency)
        if 'recency' in X.columns and 'frequency' in X.columns:
            loyalty_factor = (1 / (X['recency'] + 1)) * np.log1p(X['frequency'])
            loyalty_factor = np.clip(loyalty_factor, 0.5, 2.0)
        else:
            loyalty_factor = 1.0
        
        # Calculate duration
        base_duration = (min_duration + max_duration) / 2
        durations = base_duration * risk_factor * loyalty_factor
        durations = np.clip(durations, min_duration, max_duration)
        
        return durations.astype(int)
    
    def comprehensive_assessment(self, X: pd.DataFrame,
                                model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Perform comprehensive credit assessment
        
        Args:
            X: Customer features
            model_name: Model to use for prediction
            
        Returns:
            DataFrame with comprehensive assessment results
        """
        logger.info(f"Performing comprehensive assessment for {len(X)} customers")
        
        # Predict risk probabilities
        risk_probabilities = self.predict_risk_probability(X, model_name)
        
        # Calculate credit scores
        credit_scores = self.calculate_credit_score(risk_probabilities)
        
        # Predict loan amounts and durations
        loan_amounts = self.predict_loan_amount(X, risk_probabilities)
        loan_durations = self.predict_loan_duration(X, risk_probabilities)
        
        # Create risk categories
        risk_categories = pd.cut(
            risk_probabilities,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Loan approval recommendation
        approval_threshold = 0.7  # Adjust based on business requirements
        loan_approved = risk_probabilities < approval_threshold
        
        # Create results dataframe
        results = pd.DataFrame({
            'customer_id': X.index if 'customer_id' not in X.columns else X['customer_id'],
            'risk_probability': risk_probabilities,
            'credit_score': credit_scores,
            'risk_category': risk_categories,
            'recommended_loan_amount': loan_amounts,
            'recommended_loan_duration': loan_durations,
            'loan_approved': loan_approved,
            'assessment_date': datetime.now()
        })
        
        logger.info(f"Assessment completed. Approval rate: {loan_approved.mean():.2%}")
        
        return results
    
    def explain_prediction(self, X: pd.DataFrame, 
                          customer_index: int = 0,
                          model_name: str = 'xgboost',
                          top_features: int = 10) -> Dict[str, Any]:
        """
        Explain prediction for a specific customer
        
        Args:
            X: Customer features
            customer_index: Index of customer to explain
            model_name: Model to use
            top_features: Number of top features to show
            
        Returns:
            Dictionary with explanation
        """
        if model_name not in self.feature_importance:
            logger.warning(f"Feature importance not available for {model_name}")
            return {}
        
        # Get customer data
        customer_data = X.iloc[customer_index:customer_index+1]
        
        # Get prediction
        risk_prob = self.predict_risk_probability(customer_data, model_name)[0]
        credit_score = self.calculate_credit_score(np.array([risk_prob]))[0]
        
        # Get feature importance
        feature_imp = self.feature_importance[model_name].head(top_features)
        
        # Get customer feature values
        customer_features = {}
        for feature in feature_imp['feature']:
            if feature in customer_data.columns:
                customer_features[feature] = customer_data[feature].iloc[0]
        
        explanation = {
            'customer_id': customer_data.index[0],
            'risk_probability': risk_prob,
            'credit_score': credit_score,
            'top_features': feature_imp.to_dict('records'),
            'customer_feature_values': customer_features,
            'model_used': model_name
        }
        
        return explanation


def load_predictor(model_dir: str = 'models') -> CreditRiskPredictor:
    """
    Load and return a credit risk predictor
    
    Args:
        model_dir: Directory containing models
        
    Returns:
        CreditRiskPredictor instance
    """
    return CreditRiskPredictor(model_dir)


def main():
    """
    Main prediction function
    """
    print("Credit Risk Prediction Module")
    print("Use this module to make predictions with trained models")


if __name__ == "__main__":
    main()
