"""
Package containing model architectures for food and drink intake recognition.

This package contains various model architectures that can be used for recognizing
eating and drinking gestures from accelerometer data.

Available models:
- ignatov_cnn: 1D CNN from "Real-time human activity recognition from accelerometer data" (Ignatov, 2018)
- laura_cnn: Custom 1D CNN with 3 convolutional layers
- msconv1d: Multi-Scale 1D Convolutional Neural Network
- cnn_lstm: CNN-LSTM hybrid model
- senyurek_cnn_lstm: CNN-LSTM from "Drink Arm Snippet Detection Using IMU for Real-Time Monitoring of Drink Intake Gestures" (Senyurek et al.)
- food_drink_cnn: Optimized 1D CNN for Food and Drink Intake Recognition
- multi_input_food_drink: Multi-domain network for Food and Drink Intake Recognition
- moccia_cnn: CNN architecture from "A Novel CNN-Based Approach for Accurate and Robust Gesture Recognition" (Moccia et al.)
- moccia_cnn_lstm: Moccia CNN-LSTM architecture
"""

# Import all model creation functions to make them available at the package level
from .ignatov_cnn import create_ignatov_cnn, create_model as create_ignatov_model
from .laura_cnn import create_model as create_laura_model
from .msconv1d import create_model as create_msconv1d_model
from .cnn_lstm import create_cnn_lstm_model
from .senyurek_cnn_lstm import create_senyurek_cnn_lstm
from .food_drink_cnn import create_food_drink_cnn, create_multi_input_model
from .moccia_cnn import create_moccia_cnn, create_model as create_moccia_model
from .moccia_cnn_lstm import create_moccia_cnn_lstm as create_moccia_cnn_lstm_impl

# Create a dictionary mapping model names to their creation functions
MODEL_CREATORS = {
    'ignatov_cnn': create_ignatov_model,
    'laura_cnn': create_laura_model,
    'msconv1d': create_msconv1d_model,
    'cnn_lstm': create_cnn_lstm_model,
    'senyurek_cnn_lstm': create_senyurek_cnn_lstm,
    'food_drink_cnn': create_food_drink_cnn,
    'multi_input_food_drink': create_multi_input_model,
    'moccia_cnn': create_moccia_model,
    'moccia_cnn_lstm': create_moccia_cnn_lstm_impl,
}

def get_model_creator(model_name):
    """
    Get the model creation function for the specified model name.
    
    Args:
        model_name (str): Name of the model architecture
        
    Returns:
        callable: Function to create the specified model
        
    Raises:
        ValueError: If the specified model name is not found
    """
    if model_name not in MODEL_CREATORS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CREATORS.keys())}")
    return MODEL_CREATORS[model_name]
