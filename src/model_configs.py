"""
Model configurations for different architectures.
Add new model configurations here to easily switch between different models.
"""

from architectures.msconv1d import create_model as create_msconv1d_model
from architectures.cnn_lstm import create_cnn_lstm_model
from architectures.food_drink_cnn import create_food_drink_cnn, create_multi_input_model
from architectures.laura_cnn import create_model as create_laura_cnn
from architectures.ignatov_cnn import create_model as create_ignatov_cnn
from architectures.senyurek_cnn_lstm import create_senyurek_cnn_lstm
from architectures.moccia_cnn import create_model as create_moccia_cnn
from architectures.moccia_cnn_lstm import create_model as create_moccia_cnn_lstm

MODEL_CONFIGS = {
    'msconv1d': {
        'create_fn': create_msconv1d_model,
        'name': 'MSConv1D',
        'description': 'Multi-Scale 1D Convolutional Neural Network',
        'default_args': {}
    },
    'cnn_lstm': {
        'create_fn': create_cnn_lstm_model,
        'name': 'CNN-LSTM',
        'description': 'CNN-LSTM model from Moccia et al. (2022)',
        'default_args': {}
    },
    'food_drink_cnn': {
        'create_fn': create_food_drink_cnn,
        'name': 'Food Drink CNN',
        'description': 'Optimized 1D CNN for Food and Drink Intake Recognition',
        'default_args': {}
    },
    'laura_cnn': {
        'create_fn': create_laura_cnn,
        'name': 'Laura CNN',
        'description': '1D CNN with 3 conv layers for food/drink intake detection',
        'default_args': {}
    },
    'multi_input_food_drink': {
        'create_fn': create_multi_input_model,
        'name': 'Multi-Input Food Drink CNN',
        'description': 'Multi-domain network for Food and Drink Intake Recognition',
        'default_args': {}
    },
    'ignatov_cnn': {
        'create_fn': create_ignatov_cnn,
        'name': 'Ignatov CNN',
        'description': '1D CNN from "Real-time human activity recognition from accelerometer data" (Ignatov, 2018)',
        'default_args': {
            'l2_reg': 5e-4,
            'learning_rate': 5e-4,
            'dropout_rate': 0.5,
        }
    },
    'senyurek_cnn_lstm': {
        'create_fn': create_senyurek_cnn_lstm,
        'name': 'Senyurek CNN-LSTM',
        'description': 'CNN-LSTM model from "Drink Arm Snippet Detection Using IMU for Real-Time Monitoring of Drink Intake Gestures" (Senyurek et al.)',
        'default_args': {
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
        }
    },
    'moccia_cnn': {
        'create_fn': create_moccia_cnn,
        'name': 'Moccia CNN',
        'description': 'CNN architecture from "A Novel CNN-Based Approach for Accurate and Robust Gesture Recognition" (Moccia et al.)',
        'default_args': {
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
        }
    },
    'moccia_cnn_lstm': {
        'create_fn': create_moccia_cnn_lstm,
        'name': 'Moccia CNN-LSTM',
        'description': 'CNN-LSTM architecture based on the work by Moccia et al. with separate branches for binary and multi-class classification',
        'default_args': {
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
        }
    },
}
