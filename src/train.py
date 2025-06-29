"""
Model Training Module for Credit Risk Assessment

This module contains functions for training various credit risk models
including logistic regression, random forest, and gradient boosting models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """
    Credit Risk Model Training and Evaluation Class
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = 'default_proxy') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for model training
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features for training...")
        
        # Define feature columns (exclude non-predictive columns)
        exclude_cols = [
            'customer_id', 'segment', 'risk_category', target_col,
            'first_transaction', 'last_transaction', 'rfm_score'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        logger.info(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Logistic Regression model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model with class balancing
        model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store model and scaler
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        self.model_metrics['logistic_regression'] = metrics
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['logistic_regression'] = feature_importance
        
        logger.info(f"Logistic Regression - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest model...")
        
        # Train model with class balancing
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store model
        self.models['random_forest'] = model
        self.model_metrics['random_forest'] = metrics
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = feature_importance
        
        logger.info(f"Random Forest - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost model...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store model
        self.models['xgboost'] = model
        self.model_metrics['xgboost'] = metrics
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = feature_importance
        
        logger.info(f"XGBoost - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        return {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred),
            'accuracy': (y_true == y_pred).mean(),
            'precision': ((y_pred == 1) & (y_true == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0,
            'recall': ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0
        }
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and compare performance
        
        Args:
            X: Features
            y: Target
            test_size: Test set size
            
        Returns:
            Dictionary with all model results
        """
        logger.info("Training all models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Further split training into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        results = {}
        
        # Train models
        results['logistic_regression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        results['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        results['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        self._evaluate_on_test_set(X_test, y_test)
        
        return results
    
    def _evaluate_on_test_set(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate all models on test set
        
        Args:
            X_test: Test features
            y_test: Test target
        """
        logger.info("Evaluating models on test set...")
        
        for model_name, model in self.models.items():
            if model_name == 'logistic_regression':
                X_test_processed = self.scalers[model_name].transform(X_test)
            else:
                X_test_processed = X_test
            
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            
            test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            self.model_metrics[model_name]['test_metrics'] = test_metrics
            
            logger.info(f"{model_name} Test - AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    def save_models(self, model_dir: str = 'models'):
        """
        Save trained models and scalers
        
        Args:
            model_dir: Directory to save models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
            
            # Save scaler if exists
            if model_name in self.scalers:
                scaler_path = os.path.join(model_dir, f'{model_name}_scaler.joblib')
                joblib.dump(self.scalers[model_name], scaler_path)
                logger.info(f"Saved {model_name} scaler to {scaler_path}")
        
        # Save feature importance and metrics
        metrics_path = os.path.join(model_dir, 'model_metrics.joblib')
        joblib.dump(self.model_metrics, metrics_path)
        
        importance_path = os.path.join(model_dir, 'feature_importance.joblib')
        joblib.dump(self.feature_importance, importance_path)
        
        logger.info("All models and metadata saved successfully")


def main():
    """
    Main training function
    """
    # This would be called with actual data
    print("Credit Risk Model Training Module")
    print("Use this module to train various credit risk models")


if __name__ == "__main__":
    main()
