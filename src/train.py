import os
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from config import (
    EXPERIMENT_BASE_PATH,
    get_dataset_files,
    config,
    MANUAL_EXPERIMENT_NAME,
    get_experiment_path
)
from datasets import load_data
from transforms import preprocess_data, sliding_window
from model_configs import MODEL_CONFIGS
from config import MODEL_NAME
from model_trainer import train_model
from training_utils import train_leave_one_subject_out

def main():
    # Get model configuration
    model_config = MODEL_CONFIGS[MODEL_NAME]
    print(f"Using {model_config['name']}: {model_config['description']}")
    
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
        train_model,
        base_path=experiment_path,
        mode=config['mode'],
        sampling_rate=config['sampling_rate'],
        window_size=config['window_size'],
        overlap_size=config['overlap_size'],
        num_classes=config['num_classes']
    )
    
    # Optional: Visualize or save results
    # You can add more post-processing or visualization here
    
    return results

if __name__ == '__main__':
    main()