"""
CNN-LSTM architecture based on the paper:
"Drink Arm Snippet Detection Using IMU for Real-Time Monitoring of Drink Intake Gestures" by Senyurek et al.

Original architecture details:
- CNN part: 3 conv1d layers with 128, 64, 32 filters respectively
- Each conv layer followed by batch norm, ReLU, and max pooling
- Fully connected layer with 32 units and 50% dropout
- LSTM part: 2 LSTM layers with 64 hidden cells each
- Sequence length of 10 samples (100ms with 100Hz sampling)
- Input window: 512 samples (5.12s at 100Hz)
- Optimizer: SGD with momentum (0.9)
- Learning rate: 1e-3
- Batch size: 16
- Epochs: 3
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes=2, config=None):
    """
    Create a CNN-LSTM model based on Senyurek et al. architecture.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        num_classes (int): Number of output classes
        config (dict, optional): Configuration dictionary with model parameters
            - dropout_rate (float): Dropout rate (default: 0.5)
            
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Default configuration from the paper
    default_config = {
        'dropout_rate': 0.5,     # 50% dropout as per paper
        'learning_rate': 1e-3,   # Learning rate from paper
        'optimizer': 'sgd',      # SGD with momentum as per paper
        'momentum': 0.9,         # Momentum value from paper
        #'batch_size': 16,        # Batch size from paper
        #'num_epochs': 3,         # Epochs from paper
        #'sequence_length': 10,    # 100ms at 100Hz as per paper
        #'window_size': 512       # 5.12s at 100Hz as per paper
    }
    
    # Update default config with user provided config
    if config is not None:
        default_config.update(config)
    
    config = default_config
    
    # Input layer - expecting (timesteps, features)
    inputs = layers.Input(shape=input_shape)
    
    # Process each feature dimension separately with shared weights
    # Input shape: (batch_size, timesteps, features)
    # We need to process each feature dimension separately with the same CNN
    
    # Split the input by feature dimension
    split_layers = [layers.Lambda(lambda x: x[:, :, i:i+1])(inputs) for i in range(input_shape[1])]
    
    processed_features = []
    for i in range(input_shape[1]):
        # Add channel dimension for Conv1D
        xi = layers.Reshape((input_shape[0], 1))(split_layers[i])
        
        # CNN Block 1 - 128 filters, kernel size = 10.24s at 50Hz = 512 samples
        xi = layers.Conv1D(filters=128, 
                         kernel_size=input_shape[0]//2, 
                         padding='same')(xi)
        xi = layers.BatchNormalization()(xi)
        xi = layers.Activation('relu')(xi)
        xi = layers.MaxPooling1D(pool_size=2, strides=2)(xi)
    
        # CNN Block 2 - 64 filters
        xi = layers.Conv1D(filters=64, 
                         kernel_size=input_shape[0]//4, 
                         padding='same')(xi)
        xi = layers.BatchNormalization()(xi)
        xi = layers.Activation('relu')(xi)
        xi = layers.MaxPooling1D(pool_size=2, strides=2)(xi)
    
        # CNN Block 3 - 32 filters
        xi = layers.Conv1D(filters=32, 
                         kernel_size=input_shape[0]//8, 
                         padding='same')(xi)
        xi = layers.BatchNormalization()(xi)
        xi = layers.Activation('relu')(xi)
        xi = layers.MaxPooling1D(pool_size=2, strides=2)(xi)
        
        # Flatten the output
        xi = layers.Flatten()(xi)
        
        # Fully connected layer with 32 units and 50% dropout
        xi = layers.Dense(32, activation='relu')(xi)
        xi = layers.Dropout(0.5)(xi)
        
        # Reshape for LSTM (add time dimension)
        xi = layers.Reshape((1, 32))(xi)
        
        processed_features.append(xi)
    
    # Concatenate processed features along the time dimension
    if len(processed_features) > 1:
        x = layers.Concatenate(axis=1)(processed_features)
    else:
        x = processed_features[0]
    
    # LSTM layers with 64 hidden cells each
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    
    # Output layer - always use softmax for multiclass, sigmoid for binary
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        # Use categorical crossentropy for multiclass
        loss = 'categorical_crossentropy'
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if config['optimizer'].lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config['learning_rate'],
            momentum=config['momentum']
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    metrics = ['accuracy']
    # Only add F1 score for binary classification
    if num_classes == 2:
        from metrics import f1_score
        metrics.append(f1_score)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_senyurek_cnn_lstm(input_shape, num_classes=2, config=None):
    """
    Wrapper function to create the Senyurek CNN-LSTM model.
    This is the function that will be called by the model factory.
    """
    return create_model(input_shape, num_classes, config)
