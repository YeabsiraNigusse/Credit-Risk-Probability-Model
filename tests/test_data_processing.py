"""
Unit Tests for Data Processing Module

This module contains unit tests for the data processing functionality
including RFM analysis and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import RFMAnalyzer, FeatureEngineer


class TestRFMAnalyzer:
    """Test cases for RFM Analyzer"""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        
        # Generate sample data
        customers = [f"CUST_{i:03d}" for i in range(1, 101)]
        data = []
        
        base_date = datetime(2024, 1, 1)
        
        for customer in customers:
            # Generate random number of transactions per customer
            n_transactions = np.random.randint(5, 50)
            
            for i in range(n_transactions):
                transaction_date = base_date + timedelta(
                    days=np.random.randint(0, 365),
                    hours=np.random.randint(0, 24)
                )
                amount = np.random.uniform(10, 500)
                
                data.append({
                    'customer_id': customer,
                    'transaction_date': transaction_date,
                    'amount': amount
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def rfm_analyzer(self):
        """Create RFM analyzer instance"""
        return RFMAnalyzer(reference_date=datetime(2024, 12, 31))
    
    def test_calculate_rfm(self, rfm_analyzer, sample_transaction_data):
        """Test RFM calculation"""
        rfm_data = rfm_analyzer.calculate_rfm(sample_transaction_data)
        
        # Check that all customers are included
        assert len(rfm_data) == 100
        
        # Check that all required columns are present
        expected_columns = ['customer_id', 'recency', 'frequency', 'monetary_total', 'monetary_avg']
        assert all(col in rfm_data.columns for col in expected_columns)
        
        # Check data types and ranges
        assert rfm_data['recency'].dtype in [np.int64, np.float64]
        assert rfm_data['frequency'].dtype == np.int64
        assert rfm_data['monetary_total'].dtype == np.float64
        assert rfm_data['monetary_avg'].dtype == np.float64
        
        # Check that recency is non-negative
        assert (rfm_data['recency'] >= 0).all()
        
        # Check that frequency is positive
        assert (rfm_data['frequency'] > 0).all()
        
        # Check that monetary values are positive
        assert (rfm_data['monetary_total'] > 0).all()
        assert (rfm_data['monetary_avg'] > 0).all()
    
    def test_create_rfm_scores(self, rfm_analyzer, sample_transaction_data):
        """Test RFM score creation"""
        rfm_data = rfm_analyzer.calculate_rfm(sample_transaction_data)
        rfm_scores = rfm_analyzer.create_rfm_scores(rfm_data)
        
        # Check that score columns are added
        score_columns = ['r_score', 'f_score', 'm_score', 'rfm_score']
        assert all(col in rfm_scores.columns for col in score_columns)
        
        # Check score ranges (1-5 for quintiles)
        assert (rfm_scores['r_score'].between(1, 5)).all()
        assert (rfm_scores['f_score'].between(1, 5)).all()
        assert (rfm_scores['m_score'].between(1, 5)).all()
        
        # Check that RFM score is a string combination
        assert rfm_scores['rfm_score'].dtype == object
        assert all(len(score) == 3 for score in rfm_scores['rfm_score'])
    
    def test_create_risk_segments(self, rfm_analyzer, sample_transaction_data):
        """Test risk segment creation"""
        rfm_data = rfm_analyzer.calculate_rfm(sample_transaction_data)
        rfm_scores = rfm_analyzer.create_rfm_scores(rfm_data)
        risk_segments = rfm_analyzer.create_risk_segments(rfm_scores)
        
        # Check that segment columns are added
        segment_columns = ['segment', 'risk_category', 'default_proxy']
        assert all(col in risk_segments.columns for col in segment_columns)
        
        # Check risk categories
        valid_risk_categories = ['Low', 'Medium', 'High']
        assert risk_segments['risk_category'].isin(valid_risk_categories).all()
        
        # Check default proxy is binary
        assert risk_segments['default_proxy'].isin([0, 1]).all()
        
        # Check that High risk corresponds to default_proxy = 1
        high_risk_customers = risk_segments[risk_segments['risk_category'] == 'High']
        assert (high_risk_customers['default_proxy'] == 1).all()


class TestFeatureEngineer:
    """Test cases for Feature Engineer"""
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Create sample transaction data for testing"""
        np.random.seed(42)
        
        customers = [f"CUST_{i:03d}" for i in range(1, 21)]
        data = []
        
        base_date = datetime(2024, 1, 1)
        
        for customer in customers:
            n_transactions = np.random.randint(10, 30)
            
            for i in range(n_transactions):
                transaction_date = base_date + timedelta(
                    days=np.random.randint(0, 180),
                    hours=np.random.randint(0, 24)
                )
                amount = np.random.uniform(20, 300)
                
                data.append({
                    'customer_id': customer,
                    'transaction_date': transaction_date,
                    'amount': amount
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create Feature Engineer instance"""
        return FeatureEngineer()
    
    def test_create_transaction_features(self, feature_engineer, sample_transaction_data):
        """Test transaction feature creation"""
        features = feature_engineer.create_transaction_features(sample_transaction_data)
        
        # Check that all customers are included
        assert len(features) == 20
        
        # Check required columns
        expected_columns = [
            'customer_id', 'total_transactions', 'total_amount', 'avg_amount',
            'std_amount', 'min_amount', 'max_amount', 'customer_tenure_days',
            'transaction_velocity', 'avg_transaction_hour', 'weekend_transaction_ratio'
        ]
        assert all(col in features.columns for col in expected_columns)
        
        # Check data types and ranges
        assert features['total_transactions'].dtype == np.int64
        assert (features['total_transactions'] > 0).all()
        
        assert features['total_amount'].dtype == np.float64
        assert (features['total_amount'] > 0).all()
        
        assert features['customer_tenure_days'].dtype.kind in ['i', 'f']
        assert (features['customer_tenure_days'] >= 0).all()
        
        assert (features['weekend_transaction_ratio'].between(0, 1)).all()
    
    def test_prepare_model_data(self, feature_engineer):
        """Test model data preparation"""
        # Create sample RFM data
        rfm_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'recency': [10, 30],
            'frequency': [20, 15],
            'monetary_total': [1000, 800],
            'r_score': [4, 3],
            'f_score': [4, 3],
            'm_score': [3, 2],
            'segment': ['Champions', 'Loyal Customers'],
            'risk_category': ['Low', 'Low'],
            'default_proxy': [0, 0]
        })
        
        # Create sample transaction features
        transaction_features = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'total_transactions': [20, 15],
            'total_amount': [1000, 800],
            'avg_amount': [50, 53.33],
            'customer_tenure_days': [100, 120],
            'transaction_velocity': [0.2, 0.125]
        })
        
        model_data = feature_engineer.prepare_model_data(rfm_data, transaction_features)
        
        # Check that data is merged correctly
        assert len(model_data) == 2
        assert 'customer_id' in model_data.columns
        
        # Check that derived features are created
        assert 'rfm_combined_score' in model_data.columns
        assert 'amount_consistency' in model_data.columns
        
        # Check derived feature calculations
        expected_combined_scores = [11, 8]  # r_score + f_score + m_score
        assert model_data['rfm_combined_score'].tolist() == expected_combined_scores


class TestIntegration:
    """Integration tests for the complete data processing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data"""
        np.random.seed(42)
        
        customers = [f"CUST_{i:03d}" for i in range(1, 51)]
        data = []
        
        base_date = datetime(2024, 1, 1)
        
        for customer in customers:
            n_transactions = np.random.randint(5, 40)
            
            for i in range(n_transactions):
                transaction_date = base_date + timedelta(
                    days=np.random.randint(0, 300),
                    hours=np.random.randint(0, 24)
                )
                amount = np.random.uniform(15, 400)
                
                data.append({
                    'customer_id': customer,
                    'transaction_date': transaction_date,
                    'amount': amount
                })
        
        return pd.DataFrame(data)
    
    def test_complete_pipeline(self, sample_data):
        """Test the complete data processing pipeline"""
        # Initialize processors
        rfm_analyzer = RFMAnalyzer(reference_date=datetime(2024, 12, 31))
        feature_engineer = FeatureEngineer()
        
        # Perform RFM analysis
        rfm_data = rfm_analyzer.calculate_rfm(sample_data)
        rfm_data = rfm_analyzer.create_rfm_scores(rfm_data)
        rfm_data = rfm_analyzer.create_risk_segments(rfm_data)
        
        # Create transaction features
        transaction_features = feature_engineer.create_transaction_features(sample_data)
        
        # Combine for modeling
        model_data = feature_engineer.prepare_model_data(rfm_data, transaction_features)
        
        # Validate final dataset
        assert len(model_data) == 50
        assert 'default_proxy' in model_data.columns
        assert model_data['default_proxy'].isin([0, 1]).all()
        
        # Check that we have both low and high risk customers
        risk_distribution = model_data['risk_category'].value_counts()
        assert len(risk_distribution) > 1  # Should have multiple risk categories
        
        # Check for missing values in key columns
        key_columns = ['recency', 'frequency', 'monetary_total', 'default_proxy']
        assert not model_data[key_columns].isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__])
