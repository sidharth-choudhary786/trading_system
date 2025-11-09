import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.tsa.arima.model import ARIMA
from pykalman import KalmanFilter

class DataImputer:
    def __init__(self, method='locf'):
        self.method = method

    def impute(self, data):
        if self.method == 'locf':
            return data.ffill().bfill()
        elif self.method == 'linear':
            return data.interpolate(method='linear')
        elif self.method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            return pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
        # ... other methods
