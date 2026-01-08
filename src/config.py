import os
import glob
from datetime import datetime
from typing import Optional, Dict, Any

# Base paths
DATASET_PATH = "/home/danielgohl/Projetos/Mestrado/Datasets/grouped_data"
EXPERIMENT_BASE_PATH = "/home/danielgohl/Projetos/Mestrado/Novos_experimentos/experiments"

# Experiment Configuration
# Experiment Type: 'mc' for multiclass, 'bin' for binary
EXPERIMENT_TYPE = 'mc'  

# Model Selection
MODEL_NAME = 'wang_tcn_mha'  # Options: 'ignatov_cnn', 'laura_cnn', 'msconv1d', 'cnn_lstm'

MODEL_TYPE = 'classic'  # 'dl' para deep learning, 'classic' para modelos clássicos
CLASSIC_MODEL_NAME = 'KNN'  # Options: 'RandomForest', 'SVM', 'KNN', 'AdaBoost', 'DecisionTree', 'NaiveBayes'
#para continuar experimento adicione a raiz da pasta do experimento
#para iniciar novo experimento altere NAME = "None"
NAME = "KNN_mc_4s_50hz_2026-01-07"
#NAME = None

#alterar  linha 82 para recomeçar experimento - comentar a linha 82 e descomentar a linha 83 para usar o nome gerado automaticamente

# Configuration parameters
config = {
    # Data Processing Configuration
    "sampling_rate": 50,
    "window_size_seconds": 4,
    "overlap_fraction": 0.5,
    "columns": ['accX', 'accY', 'accZ', 'asX', 'asY', 'asZ'],
    
    # Deep Learning Training Configuration
    "num_epochs": 20,
    "batch_size": 16,
    "patience": 3,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "momentum": 0.9,
    "use_learning_rate_scheduler": False,
}

# Classic Model Hyperparameters
classic_model_config = {
    'RandomForest': {'n_estimators': 100, 'random_state': 42},
    'SVM': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'random_state': 42},
    'KNN': {'n_neighbors': 5},
    'AdaBoost': {'n_estimators': 50, 'random_state': 42},
    'DecisionTree': {'random_state': 42},
    'NaiveBayes': {},
}

# Derive mode and number of classes based on experiment type
config["mode"] = 2 if EXPERIMENT_TYPE == 'mc' else 1  #2 for multiclass, 1 for binary
NUM_CLASSES = 8 if EXPERIMENT_TYPE == 'mc' else 2  # 8 classes as specified

# Derived parameters
SAMPLING_RATE = config["sampling_rate"]
WINDOW_SIZE = int(config["window_size_seconds"] * SAMPLING_RATE)
OVERLAP_SIZE = int(WINDOW_SIZE * config["overlap_fraction"])


def get_experiment_path():
    """Get the experiment path"""
    if MANUAL_EXPERIMENT_NAME is None:
        return EXPERIMENT_BASE_PATH
    return os.path.join(EXPERIMENT_BASE_PATH, MANUAL_EXPERIMENT_NAME)

def get_dataset_files():
    """Get dataset files"""
    return sorted(glob.glob(os.path.join(DATASET_PATH, "*.pkl")))

    # Experiment Name
def get_experiment_name():
    """Generate experiment name based on configuration."""
    model_identifier = MODEL_NAME if MODEL_TYPE == 'dl' else CLASSIC_MODEL_NAME
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    if MODEL_TYPE == 'dl':
        return f"{model_identifier}_{EXPERIMENT_TYPE}_{config['window_size_seconds']}s_{config['sampling_rate']}hz_{config['optimizer']}_bs{config['batch_size']}_{timestamp}"
    else:
        return f"{model_identifier}_{EXPERIMENT_TYPE}_{config['window_size_seconds']}s_{config['sampling_rate']}hz_{timestamp}"

if NAME != None:
    MANUAL_EXPERIMENT_NAME = NAME
else: 
    MANUAL_EXPERIMENT_NAME = get_experiment_name()



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
    # Create subdirectory based on model type
    model_subdir = 'classic' if MODEL_TYPE == 'classic' else 'deep'
    exp_path = os.path.join(EXPERIMENT_BASE_PATH, model_subdir, MANUAL_EXPERIMENT_NAME)
    
    # Create required subdirectories
    os.makedirs(os.path.join(exp_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'progress'), exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'results'), exist_ok=True)
    return exp_path