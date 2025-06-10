"""
Implementation of Laura's CNN architecture for food gesture recognition.
This model uses a series of 1D convolutions with max pooling for temporal pattern recognition.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def f1_score(y_true, y_pred):
    """
    Custom F1 score metric for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        F1 score
    """
    y_pred = tf.round(y_pred)
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

def create_model(input_shape, num_classes=1, config=None):
    """
    Create and compile Laura's CNN model for food gesture recognition.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        num_classes (int): Number of output classes (default=1 for binary classification)
        config (dict, optional): Configuration dictionary with model parameters
            - learning_rate (float): Learning rate for Adam optimizer
            - dropout_rate (float): Dropout rate
            - filters (list): Number of filters for each Conv1D layer
            - kernel_size (int): Kernel size for Conv1D layers
            
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Default configuration
    default_config = {
        'learning_rate': 0.001,
        'dropout_rate': 0.1,
        'filters': [6, 6, 6],
        'kernel_size': 3
    }
    
    # Update default config with provided config
    if config is not None:
        default_config.update(config)
    
    config = default_config
    
    model = Sequential([
        # First Conv1D block
        Conv1D(
            filters=config['filters'][0],
            kernel_size=config['kernel_size'],
            activation='relu',
            input_shape=input_shape,
            padding='same'
        ),
        MaxPooling1D(),
        
        # Second Conv1D block
        Conv1D(
            filters=config['filters'][1],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same'
        ),
        MaxPooling1D(),
        
        # Third Conv1D block
        Conv1D(
            filters=config['filters'][2],
            kernel_size=config['kernel_size'],
            activation='relu',
            padding='same'
        ),
        MaxPooling1D(),
        
        # Regularization
        Dropout(config['dropout_rate']),
        
        # Output layer
        Flatten(),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    
    model.summary()
    return model

# For testing the model
def test_forward_pass():
    """Test function to verify the model can be created and process sample input."""
    import numpy as np
    
    # Create a test model
    input_shape = (100, 6)  # 100 timesteps, 6 features
    model = create_model(input_shape)
    
    # Create random test data
    x_test = np.random.rand(32, *input_shape)  # Batch of 32 samples
    
    # Test forward pass
    y_pred = model.predict(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Output range: [{y_pred.min()}, {y_pred.max()}]")

if __name__ == "__main__":
    test_forward_pass()
