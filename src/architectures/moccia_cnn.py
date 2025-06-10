"""
Implementation of the Moccia CNN architecture for gesture classification.

This architecture is based on the work:
"A Novel CNN-Based Approach for Accurate and Robust Gesture Recognition"
by Moccia et al.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
import os

# Import the f1_score metric from the metrics module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import f1_score

def create_moccia_cnn(input_shape, num_classes=2, config=None):
    """
    Implementation of the Moccia CNN architecture for gesture classification.
    
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
    
    # Layer 1: Conv1D + ReLU
    # Input: (None, 150, 6) -> Output: (None, 148, 100)
    x = layers.Conv1D(filters=100, kernel_size=3, strides=1, padding='valid', activation='relu')(input_layer)
    
    # Layer 2: Conv1D + ReLU
    # Input: (None, 148, 100) -> Output: (None, 146, 150)
    x = layers.Conv1D(filters=150, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    
    # Layer 3: Conv1D + ReLU
    # Input: (None, 146, 150) -> Output: (None, 144, 150)
    x = layers.Conv1D(filters=150, kernel_size=3, strides=1, padding='valid', activation='relu')(x)
    
    # Layer 4: Dropout
    # Input: (None, 144, 150) -> Output: (None, 144, 150)
    x = layers.Dropout(rate=dropout_rate)(x)
    
    # Layer 5: Max Pooling
    # Input: (None, 144, 150) -> Output: (None, 48, 150)
    x = layers.MaxPool1D(pool_size=3, strides=3)(x)
    
    # Layer 6: Flatten
    # Input: (None, 48, 150) -> Output: (None, 7200)
    x = layers.Flatten()(x)
    
    # Layer 7: FC + ReLU
    # Input: (None, 7200) -> Output: (None, 1000)
    x = layers.Dense(1000, activation='relu')(x)
    
    # Branch based on number of classes
    if num_classes > 2:
        # Multi-gesture classification branch
        
        # Layer 8: Dropout
        x = layers.Dropout(rate=dropout_rate)(x)
        
        # Layer 9: FC + ReLU
        x = layers.Dense(500, activation='relu')(x)
        
        # Layer 10: Dropout
        x = layers.Dropout(rate=dropout_rate)(x)
        
        # Layer 11: Output layer with softmax
        output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    else:
        # Binary classification branch
        
        # Layer 8: Dropout
        x = layers.Dropout(rate=dropout_rate)(x)
        
        # Layer 9: FC + ReLU
        x = layers.Dense(200, activation='relu')(x)
        
        # Layer 10: Dropout
        x = layers.Dropout(rate=dropout_rate)(x)
        
        # Layer 11: FC + ReLU
        x = layers.Dense(100, activation='relu')(x)
        
        # Layer 12: Dropout
        x = layers.Dropout(rate=dropout_rate)(x)
        
        # Layer 13: Output layer with softmax
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
    return create_moccia_cnn(input_shape, num_classes, config)
