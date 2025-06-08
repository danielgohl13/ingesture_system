"""
Ronao and cho model for food and drink intake detection.

This module implements a simple 1D CNN with 3 convolutional layers and max-pooling.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
)
from tensorflow.keras.metrics import Metric
import numpy as np

class F1Score(tf.keras.metrics.Metric):
    """
    Custom F1 Score metric for Keras that handles multi-class classification
    with sparse labels (not one-hot encoded).
    """
    def __init__(self, name='f1_score', num_classes=8, average='macro', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        
        # Initialize variables to track true positives, false positives, and false negatives
        self.true_positives = self.add_weight(
            'true_positives', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives', shape=(num_classes,), initializer='zeros')
        self.sample_weights = self.add_weight(
            'sample_weights', shape=(), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert sparse labels to one-hot if needed
        y_true = tf.cast(y_true, tf.int32)
        if len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Initialize updates
        tp_updates = tf.zeros((self.num_classes,), dtype=tf.float32)
        fp_updates = tf.zeros((self.num_classes,), dtype=tf.float32)
        fn_updates = tf.zeros((self.num_classes,), dtype=tf.float32)
        
        # Calculate updates for each class
        for i in range(self.num_classes):
            true_pos = tf.logical_and(
                tf.equal(y_true, i),
                tf.equal(y_pred, i)
            )
            false_pos = tf.logical_and(
                tf.not_equal(y_true, i),
                tf.equal(y_pred, i)
            )
            false_neg = tf.logical_and(
                tf.equal(y_true, i),
                tf.not_equal(y_pred, i)
            )
            
            tp_updates = tf.tensor_scatter_nd_update(
                tp_updates, [[i]], [tf.reduce_sum(tf.cast(true_pos, tf.float32))])
            fp_updates = tf.tensor_scatter_nd_update(
                fp_updates, [[i]], [tf.reduce_sum(tf.cast(false_pos, tf.float32))])
            fn_updates = tf.tensor_scatter_nd_update(
                fn_updates, [[i]], [tf.reduce_sum(tf.cast(false_neg, tf.float32))])
        
        # Update the state variables
        self.true_positives.assign_add(tp_updates)
        self.false_positives.assign_add(fp_updates)
        self.false_negatives.assign_add(fn_updates)
        
        # Update sample weights if provided
        if sample_weight is not None:
            self.sample_weights.assign_add(tf.reduce_sum(tf.cast(sample_weight, tf.float32)))
        else:
            self.sample_weights.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        
        if self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'weighted':
            weights = self.true_positives + self.false_negatives
            weights = weights / (tf.reduce_sum(weights) + tf.keras.backend.epsilon())
            return tf.reduce_sum(f1 * weights)
        else:  # micro
            precision = tf.reduce_sum(self.true_positives) / (
                tf.reduce_sum(self.true_positives) + 
                tf.reduce_sum(self.false_positives) + 
                tf.keras.backend.epsilon()
            )
            recall = tf.reduce_sum(self.true_positives) / (
                tf.reduce_sum(self.true_positives) + 
                tf.reduce_sum(self.false_negatives) + 
                tf.keras.backend.epsilon()
            )
            return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(tf.zeros((self.num_classes,), dtype=tf.float32))
        self.false_positives.assign(tf.zeros((self.num_classes,), dtype=tf.float32))
        self.false_negatives.assign(tf.zeros((self.num_classes,), dtype=tf.float32))
        self.sample_weights.assign(0.0)

def create_model(input_shape, num_classes, config=None):
    """
    Create Laura's CNN model for food/drink intake detection using Functional API.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, channels)
        num_classes (int): Number of output classes
        config (dict, optional): Configuration dictionary containing training parameters.
                               If None, default values will be used.
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'use_learning_rate_scheduler': True
        }
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First Conv + Pooling block
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D()(x)
    
    # Second Conv + Pooling block
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D()(x)
    
    # Third Conv + Pooling block
    x = Conv1D(filters=128, kernel_size=10, activation='relu')(x)
    x = MaxPooling1D()(x)
    
    # Classification head
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Configure optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = float(config.get('learning_rate', 0.001))
    
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.get(optimizer_name)
        if hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = learning_rate
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', F1Score(num_classes=num_classes, average='weighted')]
    )
    
    model.summary()
    return model


def test_model_creation():
    """Test the model creation with sample input shapes."""
    print("Testing model creation...")
    input_shape = (200, 6)  # Example input shape
    num_classes = 8  # Example number of classes
    
    # Test with default config
    model = create_model(input_shape, num_classes)
    assert model is not None, "Model creation failed"
    assert model.input_shape == (None, *input_shape), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, num_classes), f"Unexpected output shape: {model.output_shape}"
    
    # Test with custom config
    custom_config = {
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'use_learning_rate_scheduler': False
    }
    model = create_model(input_shape, num_classes, config=custom_config)
    assert model is not None, "Model creation with custom config failed"
    
    print("Model creation test passed!")


def test_forward_pass():
    """Test a forward pass through the model."""
    print("Testing forward pass...")
    input_shape = (200, 6)
    num_classes = 8
    batch_size = 2
    
    # Create a random input tensor
    x = tf.random.normal((batch_size, *input_shape))
    
    # Test with default config
    model = create_model(input_shape, num_classes)
    y = model(x)
    assert y.shape == (batch_size, num_classes), f"Unexpected output shape: {y.shape}"
    
    # Test with custom config
    custom_config = {
        'optimizer': 'rmsprop',
        'learning_rate': 0.005,
        'use_learning_rate_scheduler': True
    }
    model = create_model(input_shape, num_classes, config=custom_config)
    y = model(x)
    assert y.shape == (batch_size, num_classes), f"Unexpected output shape with custom config: {y.shape}"
    
    print("Forward pass test passed!")


if __name__ == "__main__":
    import numpy as np
    test_model_creation()
    test_forward_pass()
    print("All tests passed!")
