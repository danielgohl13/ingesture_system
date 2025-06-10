"""
Implementation of the Moccia CNN-LSTM architecture for gesture classification.

This architecture is based on the work:
"A Novel CNN-LSTM Approach for Accurate and Robust Gesture Recognition"
by Moccia et al.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
import os

# Import the f1_score metric from the metrics module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import f1_score

def create_moccia_cnn_lstm(input_shape, num_classes=2, config=None):
    """
    Implementation of the Moccia CNN-LSTM architecture for gesture classification.
    
    The architecture has two variants:
    1. Multi-gesture classification (num_classes > 2)
    2. Binary classification (num_classes = 2)
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        num_classes (int): Number of output classes (2 for binary, >2 for multi-gesture)
        config (dict, optional): Configuration dictionary with model parameters
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Get configuration parameters or use defaults
    config = config or {}
    learning_rate = config.get('learning_rate', 0.001)
    dropout_rate = config.get('dropout_rate', 0.5)
    
    # Input layer
    input_layer = layers.Input(shape=input_shape)
    
    # Add channel dimension for Conv1D
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(input_layer)
    
    # Layer 1: TimeDistributed Conv1D + ReLU with 'same' padding
    # Input: (batch, timesteps, features, 1) -> Output: (batch, timesteps, features, 100)
    x = layers.TimeDistributed(
        layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same')
    )(x)
    
    # Layer 2: TimeDistributed Conv1D + ReLU with 'same' padding
    x = layers.TimeDistributed(
        layers.Conv1D(filters=150, kernel_size=3, activation='relu', padding='same')
    )(x)
    
    # Layer 3: TimeDistributed Conv1D + ReLU with 'same' padding
    x = layers.TimeDistributed(
        layers.Conv1D(filters=150, kernel_size=3, activation='relu', padding='same')
    )(x)
    
    # Layer 4: TimeDistributed Dropout
    x = layers.TimeDistributed(layers.Dropout(rate=dropout_rate))(x)
    
    # Layer 5: TimeDistributed Max Pooling with stride 1 to maintain temporal dimension
    x = layers.TimeDistributed(
        layers.MaxPool1D(pool_size=3, strides=3, padding='same')
    )(x)
    
    # Calculate the flattened size after convolutions and pooling
    # Since we're using 'same' padding and stride=1 in maxpool, the time dimension remains the same
    flattened_size = input_shape[0] * 150  # 150 filters in the last conv layer
    
    # Layer 6: TimeDistributed Flatten
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # Reshape for LSTM: (batch, timesteps, features) where features is (input_shape[0] * 150)
    x = layers.Reshape((input_shape[0], -1))(x)
    
    # Layer 7: LSTM
    x = layers.LSTM(150, return_sequences=False)(x)
    
    # Branch based on number of classes
    if num_classes > 2:
        # Multi-gesture classification branch
        x = layers.Dense(1000, activation='relu')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Dense(500, activation='relu')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    else:
        # Binary classification branch
        x = layers.Dense(500, activation='relu')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Dense(200, activation='relu')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    if num_classes > 2:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', f1_score]
    else:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', f1_score]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_model(input_shape, num_classes=2, config=None):
    """
    Wrapper function to match the expected signature for model creation.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        num_classes (int): Number of output classes
        config (dict, optional): Configuration dictionary with model parameters
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    return create_moccia_cnn_lstm(input_shape, num_classes, config)
