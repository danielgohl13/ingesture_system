import tensorflow as tf
import numpy as np
from msconv1d import create_model

def test_model_creation():
    # Test with sample input shape
    input_shape = (100, 1)  # 100 time steps, 1 feature
    num_classes = 3
    
    # Build model
    model = create_model(input_shape, num_classes)
    
    # Check model type
    assert isinstance(model, tf.keras.Model), "Model should be a Keras Model instance"
    
    # Check input shape
    assert model.input_shape == (None, 100, 1), f"Expected input shape (None, 100, 1), got {model.input_shape}"
    
    # Check output shape
    assert model.output_shape == (None, num_classes), f"Expected output shape (None, {num_classes}), got {model.output_shape}"
    
    print("Model creation test passed!")

def test_forward_pass():
    # Create sample input
    batch_size = 2
    input_shape = (100, 1)
    num_classes = 3
    X = np.random.randn(batch_size, *input_shape).astype(np.float32)
    
    # Build model
    model = create_model(input_shape, num_classes)
    
    # Forward pass
    predictions = model.predict(X)
    
    # Check output shape
    assert predictions.shape == (batch_size, num_classes), \
        f"Expected predictions shape ({batch_size}, {num_classes}), got {predictions.shape}"
    
    # Check if outputs are probabilities (sum to 1 and between 0 and 1)
    assert np.allclose(np.sum(predictions, axis=1), 1.0), "Predictions should sum to 1"
    assert np.all(predictions >= 0) and np.all(predictions <= 1), \
        "Predictions should be between 0 and 1"
    
    print("Forward pass test passed!")

if __name__ == "__main__":
    print("Testing MSConv1D architecture...")
    test_model_creation()
    test_forward_pass()
    print("All tests passed successfully!")
