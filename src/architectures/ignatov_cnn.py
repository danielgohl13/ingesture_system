import tensorflow as tf
from tensorflow.keras import layers, Model

# Import the f1_score metric from the metrics module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import f1_score

def create_ignatov_cnn(input_shape, num_classes=2, config=None):
    """
    Implementation of the CNN architecture from:
    "Real-time human activity recognition from accelerometer data using Convolutional Neural Networks"
    by D. Ignatov, 2018.
    
    Original paper: https://www.sciencedirect.com/science/article/abs/pii/S1568494617305665
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        num_classes (int): Number of output classes
        config (dict, optional): Configuration dictionary with model parameters
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Get configuration parameters or use defaults
    config = config or {}
    l2_reg = config.get('l2_reg', 5e-4)
    learning_rate = config.get('learning_rate', 5e-4)
    dropout_rate = config.get('dropout_rate', 0.05)
    n_filters = config.get('n_filters', 196)
    filters_size = config.get('filters_size', 16)
    n_hidden = config.get('n_hidden', 1024)
    
    # Input layer
    input_layer = layers.Input(shape=input_shape)
    
    # 1D Convolutional layer
    x = layers.Conv1D(
        filters=n_filters, 
        kernel_size=filters_size, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(input_layer)
    
    # Max Pooling
    x = layers.MaxPool1D(pool_size=4)(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Fully connected layer
    x = layers.Dense(
        units=n_hidden, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    
    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer - using sigmoid for binary classification
    # For multi-class, this would need to be adjusted to softmax with num_classes units
    outputs = layers.Dense(
        units=1 if num_classes == 2 else num_classes, 
        activation='sigmoid' if num_classes == 2 else 'softmax',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Choose appropriate loss function based on number of classes
    loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', f1_score]
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
    return create_ignatov_cnn(input_shape, num_classes, config)
