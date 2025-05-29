import os
import glob
from datetime import datetime
from typing import Optional, Dict, Any

# Base paths
DATASET_PATH = "/home/danielgohl/Projetos/Mestrado/Datasets/grouped_data"
EXPERIMENT_BASE_PATH = "/home/danielgohl/Projetos/Mestrado/Novos_experimentos/experiments"

# Experiment Configuration
# Experiment Type: 'mc' for multiclass, 'bin' for binary
EXPERIMENT_TYPE = 'bin'  

# Model Selection
MODEL_NAME = 'senyurek_cnn_lstm'  # Options: 'ignatov_cnn', 'laura_cnn', 'msconv1d', 'cnn_lstm'

# Configuration parameters
config = {
    # Data Processing Configuration
    "sampling_rate": 50,  # 50Hz
    "window_size_seconds": 3,  # Janela de 3 segundos (150 amostras)
    "overlap_fraction": 0.5,  # 50% de sobreposição
    "columns": ['accX', 'accY', 'accZ', 'asX', 'asY', 'asZ'],
    
    # Training Configuration - Senyurek et al. parameters
    "num_epochs": 3,           # Número de épocas de treinamento (conforme o artigo)
    "batch_size": 16,          # Tamanho do batch (conforme o artigo)
    "patience": 3,            # Early stopping com paciência igual ao número de épocas
    "optimizer": "adam",        # SGD with momentum conforme o artigo
    "learning_rate": 0.001,    # Taxa de aprendizado do artigo
    "momentum": 0.9,           # Momentum do SGD conforme o artigo
    "use_learning_rate_scheduler": False,  # Não usar agendador de taxa de aprendizado

}

# Derive mode and number of classes based on experiment type
config["mode"] = 2 if EXPERIMENT_TYPE == 'mc' else 1  #2 for multiclass, 1 for binary
NUM_CLASSES = 8 if EXPERIMENT_TYPE == 'mc' else 2  # 8 classes as specified

# Derived parameters
SAMPLING_RATE = config["sampling_rate"]
WINDOW_SIZE = int(config["window_size_seconds"] * SAMPLING_RATE)
OVERLAP_SIZE = int(WINDOW_SIZE * config["overlap_fraction"])

#MANUAL_EXPERIMENT_NAME = f"{MODEL_NAME}_{EXPERIMENT_TYPE}_{config['optimizer']}_bs{config['batch_size']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
MANUAL_EXPERIMENT_NAME = "/home/danielgohl/Projetos/Mestrado/Novos_experimentos/experiments/senyurek_cnn_lstm_bin_sgd_bs16_2025-05-25_01-21-09"

def get_experiment_path():
    """Get the experiment path"""
    if MANUAL_EXPERIMENT_NAME is None:
        return EXPERIMENT_BASE_PATH
    return os.path.join(EXPERIMENT_BASE_PATH, MANUAL_EXPERIMENT_NAME)

def get_dataset_files():
    """Get dataset files"""
    return sorted(glob.glob(os.path.join(DATASET_PATH, "*.pkl")))

# Calculate derived parameters
config["window_size"] = int(config["window_size_seconds"] * config["sampling_rate"])
config["overlap_size"] = int(config["overlap_fraction"] * config["window_size"])
config["num_classes"] = NUM_CLASSES
config["columns_to_use"] = config["columns"] + ['label']

# Export configuration as module-level constants
SAMPLING_RATE = config["sampling_rate"]
WINDOW_SIZE_SECONDS = config["window_size_seconds"]
WINDOW_SIZE = config["window_size"]
OVERLAP_FRACTION = config["overlap_fraction"]
OVERLAP_SIZE = config["overlap_size"]
MODE = config["mode"]
NUM_CLASSES = config["num_classes"]
COLUMNS = config["columns"]
COLUMNS_TO_USE = config["columns_to_use"]

def get_dataset_files():
    """
    Retrieve all dataset files.
    
    Returns:
        list: Sorted list of dataset file paths
    """
    filenames = glob.glob(os.path.join(DATASET_PATH, "idp*.csv"))
    return sorted(filenames)

def get_experiment_path():
    """
    Get the current experiment's artifact path.
    
    Returns:
        str: Path to experiment artifacts
    """
    exp_path = os.path.join(EXPERIMENT_BASE_PATH, MANUAL_EXPERIMENT_NAME)
    # Create required subdirectories
    os.makedirs(os.path.join(exp_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'progress'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'results'), exist_ok=True)
    return exp_path
