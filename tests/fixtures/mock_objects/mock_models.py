# trading_system/tests/fixtures/mock_objects/mock_models.py
import pandas as pd
import numpy as np

class MockModel:
    """Mock ML model for testing"""
    
    def __init__(self):
        self.is_trained = False
    
    def train(self, X, y):
        """Mock training"""
        self.is_trained = True
        return {'loss': 0.1, 'accuracy': 0.8}
    
    def predict(self, X):
        """Mock prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return np.random.choice([0, 1], len(X))
    
    def save(self, path):
        """Mock save"""
        return True
    
    def load(self, path):
        """Mock load"""
        self.is_trained = True
        return True
