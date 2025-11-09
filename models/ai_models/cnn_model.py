# trading_system/models/ai_models/cnn_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import logging

from ...core.exceptions import ModelError
from ..base_model import BaseModel

class CNNModel(BaseModel):
    """
    CNN model for time series pattern recognition
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # CNN configuration
        self.sequence_length = config.get('sequence_length', 60)
        self.conv_filters = config.get('conv_filters', [64, 128, 256])
        self.kernel_sizes = config.get('kernel_sizes', [3, 3, 3])
        self.dense_units = config.get('dense_units', [128, 64])
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.logger.info("CNN model initialized")
    
    def build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build CNN model"""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        x = inputs
        
        # CNN layers
        for filters, kernel_size in zip(self.conv_filters, self.kernel_sizes):
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        
        # Global pooling and dense layers
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        for units in self.dense_units:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
        
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CNN training"""
        feature_columns = [col for col in data.columns if col != target_column and col != 'date']
        self.feature_names = feature_columns
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_sequences, y_sequences = self._create_sequences(X_scaled, y)
        
        return X_sequences, y_sequences
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """Train CNN model"""
        self.logger.info(f"Training CNN with {len(X_train)} sequences")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
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
