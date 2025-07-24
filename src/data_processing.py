"""
Data Processing Module for Credit Risk Model

This module contains functions for data preprocessing, feature engineering,
and RFM analysis for credit risk assessment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMAnalyzer:
    """
    Recency, Frequency, Monetary (RFM) Analysis for customer segmentation
    and credit risk proxy variable creation.
    """
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize RFM Analyzer
        
        Args:
            reference_date: Reference date for recency calculation. 
                          If None, uses the latest date in data.
        """
        self.reference_date = reference_date
        self.rfm_scores = None
        self.risk_segments = None
        
    def calculate_rfm(self, df: pd.DataFrame, 
                     customer_id_col: str = 'customer_id',
                     transaction_date_col: str = 'transaction_date',
                     amount_col: str = 'amount') -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer
        
        Args:
            df: Transaction dataframe
            customer_id_col: Column name for customer ID
            transaction_date_col: Column name for transaction date
            amount_col: Column name for transaction amount
            
        Returns:
            DataFrame with RFM metrics per customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Ensure date column is datetime
        df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
        
        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = df[transaction_date_col].max()
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            transaction_date_col: [
                lambda x: (self.reference_date - x.max()).days,  # Recency
                'count'  # Frequency
            ],
            amount_col: ['sum', 'mean']  # Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['recency', 'frequency', 'monetary_total', 'monetary_avg']
        rfm = rfm.reset_index()
        
        logger.info(f"RFM metrics calculated for {len(rfm)} customers")
        return rfm
    
    def create_rfm_scores(self, rfm_df: pd.DataFrame, 
                         quintiles: bool = True) -> pd.DataFrame:
        """
        Create RFM scores using quintile-based scoring
        
        Args:
            rfm_df: DataFrame with RFM metrics
            quintiles: If True, use quintiles (1-5), else use quartiles (1-4)
            
        Returns:
            DataFrame with RFM scores added
        """
        logger.info("Creating RFM scores...")
        
        n_bins = 5 if quintiles else 4
        
        # Create scores (1 = worst, 5 = best for frequency and monetary)
        # For recency, 1 = most recent (best), 5 = least recent (worst)
        rfm_df['r_score'] = pd.qcut(rfm_df['recency'], n_bins, 
                                   labels=range(n_bins, 0, -1))
        rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 
                                   n_bins, labels=range(1, n_bins + 1))
        rfm_df['m_score'] = pd.qcut(rfm_df['monetary_total'].rank(method='first'), 
                                   n_bins, labels=range(1, n_bins + 1))
        
        # Convert to numeric
        rfm_df['r_score'] = rfm_df['r_score'].astype(int)
        rfm_df['f_score'] = rfm_df['f_score'].astype(int)
        rfm_df['m_score'] = rfm_df['m_score'].astype(int)
        
        # Create combined RFM score
        rfm_df['rfm_score'] = (rfm_df['r_score'].astype(str) + 
                              rfm_df['f_score'].astype(str) + 
                              rfm_df['m_score'].astype(str))
        
        self.rfm_scores = rfm_df
        logger.info("RFM scores created successfully")
        return rfm_df
    
    def create_risk_segments(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk segments based on RFM scores as proxy for credit risk
        
        Args:
            rfm_df: DataFrame with RFM scores
            
        Returns:
            DataFrame with risk segments added
        """
        logger.info("Creating risk segments...")
        
        def segment_customers(row):
            """Segment customers based on RFM scores"""
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            
            # High-value, loyal customers (Low Risk)
            if (r >= 4) and (f >= 4) and (m >= 4):
                return 'Champions'
            elif (r >= 3) and (f >= 3) and (m >= 4):
                return 'Loyal Customers'
            elif (r >= 4) and (f >= 2) and (m >= 3):
                return 'Potential Loyalists'
            
            # Medium risk customers
            elif (r >= 4) and (f >= 1) and (m >= 1):
                return 'New Customers'
            elif (r >= 3) and (f >= 2) and (m >= 2):
                return 'Promising'
            elif (r >= 2) and (f >= 2) and (m >= 2):
                return 'Need Attention'
            
            # High risk customers
            elif (r >= 2) and (f >= 1) and (m >= 1):
                return 'About to Sleep'
            elif (r >= 1) and (f >= 2) and (m >= 1):
                return 'At Risk'
            elif (r >= 1) and (f >= 1) and (m >= 3):
                return 'Cannot Lose Them'
            else:
                return 'Lost'
        
        rfm_df['segment'] = rfm_df.apply(segment_customers, axis=1)
        
        # Create binary risk classification (proxy for default)
        low_risk_segments = ['Champions', 'Loyal Customers', 'Potential Loyalists']
        medium_risk_segments = ['New Customers', 'Promising', 'Need Attention']
        high_risk_segments = ['About to Sleep', 'At Risk', 'Cannot Lose Them', 'Lost']
        
        rfm_df['risk_category'] = rfm_df['segment'].apply(
            lambda x: 'Low' if x in low_risk_segments 
                     else 'Medium' if x in medium_risk_segments 
                     else 'High'
        )
        
        # Create binary target variable (0 = Good, 1 = Bad)
        rfm_df['default_proxy'] = (rfm_df['risk_category'] == 'High').astype(int)
        
        self.risk_segments = rfm_df
        logger.info(f"Risk segments created. Distribution: \n{rfm_df['risk_category'].value_counts()}")
        return rfm_df


class FeatureEngineer:
    """
    Feature engineering class for credit risk modeling
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction-based features for credit risk modeling
        
        Args:
            df: Transaction dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating transaction features...")
        
        # Ensure date column is datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Time-based features
        df['transaction_hour'] = df['transaction_date'].dt.hour
        df['transaction_day'] = df['transaction_date'].dt.day
        df['transaction_month'] = df['transaction_date'].dt.month
        df['transaction_weekday'] = df['transaction_date'].dt.weekday
        df['is_weekend'] = (df['transaction_weekday'] >= 5).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        
        # Customer aggregated features
        customer_features = df.groupby('customer_id').agg({
            'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'transaction_date': ['min', 'max'],
            'transaction_hour': 'mean',
            'is_weekend': 'mean'
        }).round(2)
        
        # Flatten column names
        customer_features.columns = [
            'total_transactions', 'total_amount', 'avg_amount', 'std_amount',
            'min_amount', 'max_amount', 'first_transaction', 'last_transaction',
            'avg_transaction_hour', 'weekend_transaction_ratio'
        ]
        
        # Calculate customer tenure
        customer_features['customer_tenure_days'] = (
            customer_features['last_transaction'] - customer_features['first_transaction']
        ).dt.days
        
        # Transaction velocity (transactions per day)
        customer_features['transaction_velocity'] = (
            customer_features['total_transactions'] / 
            (customer_features['customer_tenure_days'] + 1)
        )
        
        customer_features = customer_features.reset_index()
        logger.info(f"Transaction features created for {len(customer_features)} customers")
        return customer_features
    
    def prepare_model_data(self, rfm_data: pd.DataFrame, 
                          transaction_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combine RFM data with transaction features for modeling
        
        Args:
            rfm_data: RFM analysis results
            transaction_features: Transaction-based features
            
        Returns:
            Combined dataset ready for modeling
        """
        logger.info("Preparing model data...")
        
        # Merge datasets
        model_data = rfm_data.merge(transaction_features, on='customer_id', how='inner')
        
        # Handle missing values
        numeric_columns = model_data.select_dtypes(include=[np.number]).columns
        model_data[numeric_columns] = model_data[numeric_columns].fillna(0)
        
        # Create additional derived features
        model_data['rfm_combined_score'] = (
            model_data['r_score'] + model_data['f_score'] + model_data['m_score']
        )
        
        model_data['amount_consistency'] = (
            model_data['std_amount'] / (model_data['avg_amount'] + 1)
        )
        
        logger.info(f"Model data prepared with {len(model_data)} customers and {len(model_data.columns)} features")
        return model_data


def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess transaction data
    
    Args:
        file_path: Path to the transaction data file
        
    Returns:
        Tuple of (processed_data, feature_data)
    """
    logger.info(f"Loading data from {file_path}")
    
    # Load data (assuming CSV format)
    df = pd.read_csv(file_path)
    
    # Initialize processors
    rfm_analyzer = RFMAnalyzer()
    feature_engineer = FeatureEngineer()
    
    # Perform RFM analysis
    rfm_data = rfm_analyzer.calculate_rfm(df)
    rfm_data = rfm_analyzer.create_rfm_scores(rfm_data)
    rfm_data = rfm_analyzer.create_risk_segments(rfm_data)
    
    # Create transaction features
    transaction_features = feature_engineer.create_transaction_features(df)
    
    # Combine for modeling
    model_data = feature_engineer.prepare_model_data(rfm_data, transaction_features)
    
    return model_data, rfm_data


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module for Credit Risk Model")
    print("This module provides RFM analysis and feature engineering capabilities")
