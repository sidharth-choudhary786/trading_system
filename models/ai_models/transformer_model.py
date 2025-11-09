# trading_system/models/ai_models/transformer_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler

from ...core.exceptions import ModelError
from ..base_model import BaseModel

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
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("Transformer model initialized")
    
    def _transformer_encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """Transformer encoder layer"""
        # Positional encoding
        x = self._positional_encoding(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = self._transformer_block(x)
        
        return x
    
    def _positional_encoding(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to inputs"""
        seq_len = tf.shape(inputs)[1]
        d_model = inputs.shape[-1]
        
        # Create positional encoding matrix
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                         -(np.log(10000.0) / d_model))
        
        pos_encoding = tf.zeros((seq_len, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len)[:, tf.newaxis] for _ in range(d_model // 2)], axis=-1),
            tf.sin(position * div_term)
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len)[:, tf.newaxis] for _ in range(d_model // 2, d_model)], axis=-1),
            tf.cos(position * div_term)
        )
        
        return inputs + pos_encoding[tf.newaxis, :, :]
    
    def _transformer_block(self, x: tf.Tensor) -> tf.Tensor:
        """Single transformer block"""
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(x, x)
        attn_output = tf.keras.layers.Dropout(self.dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward network
        ffn_output = tf.keras.layers.Dense(self.dff, activation='relu')(x)
        ffn_output = tf.keras.layers.Dense(self.d_model)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(self.dropout_rate)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        return x
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build transformer model"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Feature projection
        x = tf.keras.layers.Dense(self.d_model)(inputs)
        
        # Transformer encoder
        x = self._transformer_encoder(x)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for transformer training"""
        feature_columns = [col for col in data.columns if col != target_column and col != 'date']
        self.feature_names = feature_columns
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_sequences, y_sequences = self._create_sequences(X_scaled, y)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for transformer"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """Train transformer model"""
        self.logger.info(f"Training Transformer with {len(X_train)} sequences")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            validation_split=0.2 if X_val is None else 0.0,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'epochs_trained': len(history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if not self.is_trained:
            raise ModelError("Model not trained")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
