import tensorflow as tf

def build_multiscale_conv1d(input_shape, num_classes):
    """Build a multi-scale 1D CNN model.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Compiled model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Initial Convolution and Pooling
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Multi-Scale Convolutional Block
    branch1 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    branch2 = tf.keras.layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    branch3 = tf.keras.layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.concatenate([branch1, branch2, branch3], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # Additional Convolution and Pooling
    x = tf.keras.layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Ensure output shape matches num_classes
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_model(input_shape, num_classes, config=None):
    """Create and compile the MS-Conv1D model.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, n_features)
        num_classes (int): Number of output classes
        config (dict, optional): Configuration dictionary containing training parameters.
                               If None, default values will be used.
        
    Returns:
        tf.keras.Model: Compiled MS-Conv1D model
    """
    if config is None:
        config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'use_learning_rate_scheduler': True
        }
    
    # Build the model
    model = build_multiscale_conv1d(input_shape, num_classes)
    
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
    
    # Choose the appropriate loss function based on number of classes
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"Model compiled with {loss} loss and {metrics} metrics")
    print(f"Input shape: {input_shape}, Output classes: {num_classes}")
    
    return model