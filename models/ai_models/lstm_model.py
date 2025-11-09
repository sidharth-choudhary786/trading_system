# trading_system/models/ai_models/lstm_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ...core.exceptions import ModelError
from ..base_model import BaseModel

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
        self.patience = config.get('patience', 10)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("LSTM model initialized")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        """
        if data.empty:
            raise ModelError("Empty data provided for training")
        
        # Extract features and target
        feature_columns = [col for col in data.columns if col != target_column and col != 'date']
        self.feature_names = feature_columns
        
        if not feature_columns:
            raise ModelError("No feature columns found in data")
        
        # Prepare features
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
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
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(25, activation='relu'))
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the LSTM model
        """
        if X_train is None or len(X_train) == 0:
            raise ModelError("No training data provided")
        
        self.logger.info(f"Training LSTM model with {len(X_train)} sequences")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2
            )
        ]
        
        # Train model
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            validation_split=self.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )
        
        self.is_trained = True
        
        # Return training history
        return {
            'history': history.history,
            'epochs_trained': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history.get('val_loss', [None])[-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_trained or self.model is None:
            raise ModelError("Model not trained")
        
        if X is None or len(X) == 0:
            raise ModelError("No data provided for prediction")
        
        # Ensure correct sequence length
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_sequences(self, data: pd.DataFrame, steps: int = 1) -> np.ndarray:
        """Predict multiple steps ahead"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        predictions = []
        current_sequence = data[self.feature_names].values[-self.sequence_length:]
        
        for _ in range(steps):
            # Scale current sequence
            current_scaled = self.scaler.transform(current_sequence)
            current_scaled = current_scaled.reshape(1, self.sequence_length, len(self.feature_names))
            
            # Predict next value
            next_pred = self.predict(current_scaled)[0]
            predictions.append(next_pred)
            
            # Update sequence (using predicted value)
            new_row = current_sequence[-1].copy()
            # Update the target feature (assuming it's the first feature)
            new_row[0] = next_pred  
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((predictions - y_test) / y_test)) * 100
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'r2': 1 - (mse / np.var(y_test))
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance (LSTM doesn't have direct feature importance)"""
        if not self.is_trained:
            return {}
        
        # For LSTM, we can use permutation importance or other methods
        return {
            'message': 'Feature importance not directly available for LSTM. Use permutation importance.',
            'feature_names': self.feature_names
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_trained:
            raise ModelError("No trained model to save")
        
        # Save TensorFlow model
        self.model.save(filepath)
        
        # Save scaler and configuration
        import joblib
        config_data = {
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        joblib.dump(config_data, filepath + '_config.pkl')
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        # Load TensorFlow model
        self.model = tf.keras.models.load_model(filepath)
        
        # Load configuration
        import joblib
        config_data = joblib.load(filepath + '_config.pkl')
        
        self.sequence_length = config_data['sequence_length']
        self.feature_names = config_data['feature_names']
        self.scaler = config_data['scaler']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")
