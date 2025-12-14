"""
Fake News Detection System
PhD-level machine learning project for fake news detection
"""

__version__ = "1.0.0"
__author__ = "Harshit Dubey"

from src.data_loader import DataLoader
from src.preprocessing import AdvancedPreprocessor
from src.feature_engineering import FeatureEngineer

__all__ = ['DataLoader', 'AdvancedPreprocessor', 'FeatureEngineer']