import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ----------------------------
# Custom Transformers
# ----------------------------

class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['Hour'] = X[self.datetime_col].dt.hour
        X['Day']   = X[self.datetime_col].dt.day
        X['Month'] = X[self.datetime_col].dt.month
        X['Year']  = X[self.datetime_col].dt.year
        return X

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = (
            X.groupby('AccountId')['Amount']
             .agg([('TotalAmount','sum'),
                   ('AvgAmount','mean'),
                   ('TransactionCount','count'),
                   ('StdAmount','std')])
             .reset_index()
        )
        X = X.merge(agg, on='AccountId', how='left')
        return X

# ----------------------------
# Final Pipeline builder
# ----------------------------

def build_pipeline():
    numeric_features = [
        'Amount', 'Value',
        'TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount',
        'Hour', 'Day', 'Month', 'Year'
    ]
    categorical_features = [
        'ProductCategory',
        'CurrencyCode',
        'CountryCode',
        'ProviderId',
        'ChannelId',
        'PricingStrategy'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    full_pipeline = Pipeline(steps=[
        ('time_features', TimeFeaturesExtractor()),
        ('aggregate', AggregateFeatures()),
        ('preproc', preprocessor)
    ])

    return full_pipeline
