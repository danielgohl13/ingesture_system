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
            'n_filters': 196,
            'filters_size': 16,
            'n_hidden': 1024
        }
    },
    'senyurek_cnn_lstm': {
        'create_fn': create_senyurek_cnn_lstm,
        'name': 'Senyurek CNN-LSTM',
        'description': 'CNN-LSTM from "Drink Arm Snippet Detection Using IMU for Real-Time Monitoring of Drink Intake Gestures" (Senyurek et al.)',
        'default_args': {
            'dropout_rate': 0.5,      # 50% dropout as per paper
            'learning_rate': 0.001,   # Learning rate from paper
            'optimizer': 'sgd',       # SGD with momentum as per paper
            'momentum': 0.9,          # Momentum value from paper
            'batch_size': 16,         # Batch size from paper
            'num_epochs': 3,          # Epochs from paper
            #'sequence_length': 10,    # 100ms at 100Hz as per paper
            #'window_size': 512,       # 5.12s at 100Hz as per paper
            #'sampling_rate': 100      # 100Hz as per paper
        }
    }
}
