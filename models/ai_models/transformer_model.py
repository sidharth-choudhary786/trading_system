# trading_system/models/ai_models/transformer_model.py
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

class TransformerModel(BaseModel):
    """
    Transformer model for time series forecasting
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Transformer configuration
        self.sequence_length = config.get('sequence_length', 60)
        self.d_model = config.get('d_model', 64)
        self.num_heads = config.get('num_heads', 4)
        self.dff = config.get('dff', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("Transformer model initialized")
    
    def _transformer_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transformer encoder implementation"""
        # Positional encoding
        seq_len = tf.shape(inputs)[1]
        d_model = inputs.shape[-1]
        
        # Create positional encoding
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        angle_rads = np.arange(seq_len)[:, np.newaxis] * angle_rates
        
        # Apply sine to even indices, cosine to odd indices
        pos_encoding = np.zeros(angle_rads.shape)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        
        # Add positional encoding to input
        inputs += pos_encoding
        
        # Dropout
        inputs = tf.keras.layers.Dropout(self.dropout_rate)(inputs)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout_rate
            )(inputs, inputs)
            
            # Add & Norm
            attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                inputs + attention_output
            )
            
            # Feed forward
            ffn_output = tf.keras.Sequential([
                tf.keras.layers.Dense(self.dff, activation='relu'),
                tf.keras.layers.Dense(self.d_model),
                tf.keras.layers.Dropout(self.dropout_rate)
            ])(attention_output)
            
            # Add & Norm
            inputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                attention_output + ffn_output
            )
        
        return inputs
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build transformer model"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Project input to d_model dimensions
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Transformer encoder
        x = self._transformer_encoder(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for transformer training"""
        if len(X) != len(y):
            raise ModelError("X and y must have same length")
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y.values)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for transformer"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the transformer model"""
        try:
            self.logger.info("Starting Transformer model training")
            
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
            
            self.logger.info("Transformer model training completed")
            
        except Exception as e:
            raise ModelError(f"Transformer training failed: {e}")
    
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
            raise ModelError(f"Transformer prediction failed: {e}")
    
    def _create_prediction_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for prediction"""
        sequences = []
        
        for i in range(self.sequence_length, len(X) + 1):
            sequences.append(X[i-self.sequence_length:i])
        
        return np.array(sequences)
    
    def save_model(self, path: str):
        """Save transformer model"""
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
                    'd_model': self.d_model,
                    'num_heads': self.num_heads
                }
            }, f"{path}_metadata.pkl")
            
            self.logger.info(f"Transformer model saved to {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to save transformer model: {e}")
    
    def load_model(self, path: str):
        """Load transformer model"""
        try:
            self.model = tf.keras.models.load_model(f"{path}_model.h5")
            
            import joblib
            metadata = joblib.load(f"{path}_metadata.pkl")
            
            self.scaler = metadata['scaler']
            self.feature_names = metadata['feature_names']
            
            config = metadata['config']
            self.sequence_length = config['sequence_length']
            self.d_model = config['d_model']
            self.num_heads = config['num_heads']
            
            self.is_trained = True
            self.logger.info(f"Transformer model loaded from {path}")
            
        except Exception as e:
            raise ModelError(f"Failed to load transformer model: {e}")
