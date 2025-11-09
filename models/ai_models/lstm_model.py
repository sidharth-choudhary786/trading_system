# trading_system/models/ai_models/lstm_model.py
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

class LSTMModel(BaseModel):
    """
    LSTM Neural Network for time series forecasting
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # LSTM configuration
        self.sequence_length = config.get('sequence_length', 60)
        self.lstm_units = config.get('lstm_units', [50, 25])
        self.dropout_rate = config.get('dropout_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("LSTM model initialized")
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        if len(X) != len(y):
            raise ModelError("X and y must have same length")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y.values)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = tf.keras.Sequential()
        
        # First LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1,
            input_shape=input_shape,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_sequences = i < len(self.lstm_units[1:]) - 1
            model.add(tf.keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
        
        # Dense layers
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
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
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the LSTM model"""
        try:
            self.logger.info("Starting LSTM model training")
            
            # Prepare data
            X_sequences, y_sequences = self.prepare_data(X, y)
            
            # Build model
            input_shape = (X_sequences.shape[1], X_sequences.shape[2])
            self.model = self.build_model(input_shape)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_sequences,
                y_sequences,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important for time series
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            self.logger.info("LSTM model training completed")
            
        except Exception as e:
            raise ModelError(f"LSTM training failed: {e}")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions"""
        if not self.is_trained:
            raise ModelError("Model must be trained before prediction")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create sequences for prediction
            X_sequences = self._create_prediction_sequences(X_scaled)
            
            # Generate predictions
            predictions = self.model.predict(X_sequences, verbose=0)
            predictions = predictions.flatten()
            
            # Align with original index
            start_idx = self.sequence_length
            end_idx = len(X)
            aligned_predictions = np.full(len(X), np.nan)
            aligned_predictions[start_idx:end_idx] = predictions[:end_idx-start_idx]
            
            return pd.Series(aligned_predictions, index=X.index)
            
        except Exception as e:
            raise ModelError(f"LSTM prediction failed: {e}")
    
    def _create_prediction_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for prediction"""
        sequences = []
        
        for i in range(self.sequence_length, len(X) + 1):
            sequences.append(X[i-self.sequence_length:i])
        
        return np.array(sequences)
    
    def save_model(self, path: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        try:
            # Save TensorFlow model
            self.model.save(f"{path}_model.h5")
            
            # Save scaler and configuration
            import joblib
            joblib.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': {
                    'sequence_length': self.sequence_length,
                    'lstm_units': self.lstm_units,
                    'dropout_rate': self.dropout_rate
                }
            }, f"{path}_metadata.pkl")
            
            self.logger.info(f"LSTM model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save LSTM model: {e}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        try:
            # Load TensorFlow model
            self.model = tf.keras.models.load_model(f"{path}_model.h5")
            
            # Load metadata
            import joblib
            metadata = joblib.load(f"{path}_metadata.pkl")
            
            self.scaler = metadata['scaler']
            self.feature_names = metadata['feature_names']
            
            # Update configuration
            config = metadata['config']
            self.sequence_length = config['sequence_length']
            self.lstm_units = config['lstm_units']
            self.dropout_rate = config['dropout_rate']
            
            self.is_trained = True
            self.logger.info(f"LSTM model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load LSTM model: {e}")
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance (LSTM doesn't have native feature importance)"""
        # For LSTM, we can use permutation importance or other methods
        # This is a simplified version
        if not self.feature_names:
            return pd.Series()
        
        # Return equal importance for all features (placeholder)
        importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        return pd.Series(importance, index=self.feature_names)
