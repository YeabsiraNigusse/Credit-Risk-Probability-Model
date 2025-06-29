"""
Credit Risk Probability Model Package

This package contains modules for building and deploying credit risk models
for Bati Bank's buy-now-pay-later service.
"""

__version__ = "1.0.0"
__author__ = "Bati Bank Analytics Team"
__email__ = "analytics@batibank.com"

from . import data_processing, train, predict

__all__ = ["data_processing", "train", "predict"]
