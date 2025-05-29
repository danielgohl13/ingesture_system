"""
Implementation of the Food and Drink Intake Recognition System Architecture

Based on the paper's deep learning approach for recognizing eating and drinking 
gestures using wrist-worn accelerometer data.

Key Components:
- 1D CNN architecture optimized for gesture recognition
- Supports multi-class classification
- Configurable for different input shapes and number of classes
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
)

def create_food_drink_cnn(input_shape, num_classes=3):
    """
    Create the optimized 1D CNN architecture for gesture recognition.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, num_channels)
        num_classes (int): Number of output classes (default: 3 for eating, drinking, null)
    
    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First Convolutional Layer
    x = Conv1D(
        filters=16,  # Optimal filter count from the paper
        kernel_size=25,  # Optimal kernel size 
        activation='relu',
        padding='same'
    )(inputs)
    x = BatchNormalization()(x)
    
    # Second Convolutional Layer
    x = Conv1D(
        filters=16, 
        kernel_size=25, 
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    
    # Third Convolutional Layer
    x = Conv1D(
        filters=16, 
        kernel_size=25, 
        activation='relu',
        padding='same'
    )(x)
    x = BatchNormalization()(x)
    
    # Max Pooling
    x = MaxPooling1D(pool_size=2, strides=1)(x)
    
    # Flatten
    x = Flatten()(x)
    
    # Fully Connected Layer with Dropout
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add a label conversion method for consistency
    def convert_labels(y):
        """
        Convert labels to appropriate format for the model.
        
        Args:
            y (numpy.ndarray): Input labels
        
        Returns:
            numpy.ndarray: Converted labels
        """
        # If labels are already one-hot or in correct format, return as-is
        if len(y.shape) > 1 or y.max() >= num_classes:
            return y
        return y
    
    model.convert_labels = convert_labels
    
    return model

# Optional: Additional model variants can be added here
def create_multi_input_model(raw_input_shape, spec_input_shape, mtf_input_shape, num_classes=3):
    """
    Multi-input model combining different signal representations.
    
    Args:
        raw_input_shape (tuple): Shape of raw accelerometer data
        spec_input_shape (tuple): Shape of spectrogram input
        mtf_input_shape (tuple): Shape of Markov Transition Field input
        num_classes (int): Number of output classes
    
    Returns:
        Multi-input Keras model
    """
    # Raw Accelerometer Input Branch
    raw_input = Input(shape=raw_input_shape)
    raw_branch = create_food_drink_cnn(raw_input_shape, num_classes=num_classes)(raw_input)
    
    # Spectrogram Input Branch
    spec_input = Input(shape=spec_input_shape)
    spec_branch = tf.keras.applications.MobileNetV2(
        input_shape=spec_input_shape, 
        include_top=False, 
        weights=None
    )(spec_input)
    spec_branch = Flatten()(spec_branch)
    spec_branch = Dense(100, activation='relu')(spec_branch)
    
    # Markov Transition Field Input Branch
    mtf_input = Input(shape=mtf_input_shape)
    mtf_branch = tf.keras.applications.MobileNetV2(
        input_shape=mtf_input_shape, 
        include_top=False, 
        weights=None
    )(mtf_input)
    mtf_branch = Flatten()(mtf_branch)
    mtf_branch = Dense(100, activation='relu')(mtf_branch)
    
    # Concatenate branches
    merged = tf.keras.layers.concatenate([raw_branch, spec_branch, mtf_branch])
    
    # Final classification layers
    x = Dense(100, activation='relu')(merged)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create multi-input model
    model = Model(
        inputs=[raw_input, spec_input, mtf_input], 
        outputs=outputs
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
