"""
Implementation of CNN-LSTM model proposed by Moccia et al. (2022) for drinking gesture 
classification using a wearable wrist sensor.

Reference: Moccia et al. "Towards monitoring medical adherence using a wristband 
and machine learning", Computer Methods and Programs in Biomedicine, 2022.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
)

def create_cnn_lstm_model(input_shape, num_classes=2):
    """
    Creates the CNN-LSTM model architecture based on the original design.
    
    Args:
        input_shape: Shape of input data (window_size, num_channels)
        num_classes: Number of output classes (2 for binary, 8 for multiclass)
        
    Returns:
        model: Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First Convolutional Layer
    x = Conv1D(filters=100, kernel_size=3, activation='relu')(inputs)
    
    # Second Convolutional Layer
    x = Conv1D(filters=150, kernel_size=3, activation='relu')(x)
    
    # Third Convolutional Layer
    x = Conv1D(filters=150, kernel_size=3, activation='relu')(x)
    
    # Dropout
    x = Dropout(0.5)(x)
    
    # Max Pooling
    x = MaxPooling1D(pool_size=3, strides=3)(x)
    
    # Flatten
    x = tf.keras.layers.Flatten()(x)
    
    # Dense Layer for Feature Representation
    x = Dense(1000, activation='relu')(x)
    
    # LSTM Layer
    x = tf.keras.layers.RepeatVector(1)(x)
    x = LSTM(units=150, return_sequences=False)(x)
    
    # Classification Layers
    if num_classes == 2:
        # Binary Classification Architecture
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Use softmax for binary classification to ensure class probabilities
        outputs = Dense(2, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    else:
        # Multi-class Classification Architecture
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    
    def convert_labels(y):
        # Convert labels to one-hot encoding for binary classification
        if num_classes == 2:
            if len(y.shape) == 1 or y.shape[1] == 1:
                return tf.keras.utils.to_categorical(y, num_classes=2)
            return y
        return y
    
    # Wrap the model to convert labels
    model.convert_labels = convert_labels
    
    return model
