# tests/test_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import split_data, evaluate_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def test_split_data_shapes():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, size=100))
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == 100
    assert len(y_train) + len(y_test) == 100

def test_evaluate_model_keys():
    model = RandomForestClassifier().fit(np.random.rand(50, 3), np.random.randint(0, 2, size=50))
    metrics = evaluate_model(model, np.random.rand(10, 3), np.random.randint(0, 2, size=10))
    assert all(k in metrics for k in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])
