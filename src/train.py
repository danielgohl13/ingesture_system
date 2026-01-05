import os
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from config import (
    EXPERIMENT_BASE_PATH,
    get_dataset_files,
    config,
    MANUAL_EXPERIMENT_NAME,
    get_experiment_path,
    MODEL_NAME,
    MODEL_TYPE,
    CLASSIC_MODEL_NAME,
    EXPERIMENT_TYPE
)
from datasets import load_data
from transforms import preprocess_data, sliding_window
from model_configs import MODEL_CONFIGS
from model_trainer import train_model
from training_utils import train_leave_one_subject_out

def log_experiment_config():
    """Logs the main configuration parameters of the experiment."""
    print("="*50)
    print(" " * 15 + "EXPERIMENT CONFIGURATION")
    print("="*50)
    
    print(f"Experiment Type: {EXPERIMENT_TYPE}")
    print(f"Model Type: {MODEL_TYPE}")

    if MODEL_TYPE == 'dl':
        model_config = MODEL_CONFIGS[MODEL_NAME]
        print(f"DL Model: {model_config['name']}")
    else:
        print(f"Classic Model: {CLASSIC_MODEL_NAME}")

    print("-" * 50)
    print("Data Parameters:")
    print(f"  Sampling Rate: {config.get('sampling_rate')} Hz")
    print(f"  Window Size: {config.get('window_size_seconds')} seconds ({config.get('window_size')} samples)")
    print(f"  Overlap: {config.get('overlap_fraction')*100}% ({config.get('overlap_size')} samples)")
    print(f"  Number of Classes: {config.get('num_classes')}")
    
    if MODEL_TYPE == 'dl':
        print("-" * 50)
        print("DL Training Parameters:")
        print(f"  Epochs: {config.get('num_epochs')}")
        print(f"  Batch Size: {config.get('batch_size')}")
        print(f"  Optimizer: {config.get('optimizer')}")
        print(f"  Learning Rate: {config.get('learning_rate')}")

    print("="*50)

def main():
    # Log the configuration
    log_experiment_config()
    
    # Get experiment name and path from config
    experiment_name = MANUAL_EXPERIMENT_NAME
    experiment_path = get_experiment_path()
    
    print(f"Starting experiment: {experiment_name}")
    
    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'progress'), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'results'), exist_ok=True)
    
    # Get dataset files
    filenames = get_dataset_files()
    
    # Perform training with progress tracking
    results = train_leave_one_subject_out(
        filenames,
        load_data,
        preprocess_data,
        sliding_window,
        train_model,  # This is passed but only used for DL models
        base_path=experiment_path,
        mode=config['mode'],
        sampling_rate=config['sampling_rate'],
        window_size=config['window_size'],
        overlap_size=config['overlap_size'],
        num_classes=config['num_classes']
    )
    
    print("Training finished.")
    return results

if __name__ == '__main__':
    main()