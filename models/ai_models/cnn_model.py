# trading_system/models/ai_models/cnn_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..base_model import BaseModel
from ...core.exceptions import ModelError

class CNNModel(BaseModel):
    """
    CNN model for time series forecasting with 1D convolutions
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # CNN configuration
        self.sequence_length = config.get('sequence_length', 60)
        self.filters = config.get('filters', [64, 32, 16])
        self.kernel_sizes = config.get('kernel_sizes', [3, 3, 3])
        self.pool_sizes = config.get('pool_sizes', [2, 2, 2])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("CNN model initialized")
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build CNN model architecture"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # CNN layers
        for i, (filters, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            model.add(tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            ))
            
            # Add pooling layer
            if i < len(self.pool_sizes):
                model.add(tf.keras.layers.MaxPooling1D(
                    pool_size=self.pool_sizes[i]
                ))
            
            # Add dropout
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Flatten and dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CNN training"""
        if len(X) != len(y):
            raise ModelError("X and y must have same length")
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y.values)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the CNN model"""
        try:
            self.logger.info("Starting CNN model training")
            
            # Prepare data
            X_sequences, y_sequences = self.prepare_data(X, y)
            
            # Build model
            input_shape = (X_sequences.shape[1], X_sequences.shape[2])
            self.model = self.build_model(input_shape)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_sequences,
                y_sequences,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1,
                shuffle=False
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            self.logger.info("CNN model training completed")
            
        except Exception as e:
            raise ModelError(f"CNN training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions"""
        if not self.is_trained:
            raise ModelError("Model must be trained before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            X_sequences = self._create_prediction_sequences(X_scaled)
            
            predictions = self.model.predict(X_sequences, verbose=0).flatten()
            
            # Align predictions
            start_idx = self.sequence_length
            end_idx = len(X)
            aligned_predictions = np.full(len(X), np.nan)
            aligned_predictions[start_idx:end_idx] = predictions[:end_idx-start_idx]
            
            return pd.Series(aligned_predictions, index=X.index)
            
        except Exception as e:
            raise ModelError(f"CNN prediction failed: {e}")
    
    def _create_prediction_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for prediction"""
        sequences = []
        
        for i in range(self.sequence_length, len(X) + 1):
            sequences.append(X[i-self.sequence_length:i])
        
        return np.array(sequences)
    
    def save_model(self, path: str):
        """Save CNN model"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            self.model.save(f"{path}_model.h5")
            
            import joblib
            joblib.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': {
                    'sequence_length': self.sequence_length,
                    'filters': self.filters,
                    'kernel_sizes': self.kernel_sizes
                }
            }, f"{path}_metadata.pkl")
            
            self.logger.info(f"CNN model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save CNN model: {e}")
    
    def load_model(self, path: str):
        """Load CNN model"""
        try:
            self.model = tf.keras.models.load_model(f"{path}_model.h5")
            
            import joblib
            metadata = joblib.load(f"{path}_metadata.pkl")
            
            self.scaler = metadata['scaler']
            self.feature_names = metadata['feature_names']
            
            config = metadata['config']
            self.sequence_length = config['sequence_length']
            self.filters = config['filters']
            self.kernel_sizes = config['kernel_sizes']
            
            self.is_trained = True
            self.logger.info(f"CNN model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load CNN model: {e}")
