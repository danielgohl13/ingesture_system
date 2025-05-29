import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from utils import carregar_progresso, salvar_progresso

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """
    Plot a confusion matrix using seaborn.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        labels (list): List of class labels
        title (str, optional): Title of the plot
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

from model_configs import MODEL_CONFIGS
from config import MODEL_NAME, config
import json
import os

def save_training_config(base_path, config, model_name):
    """Save training configuration to a file.
    
    Args:
        base_path (str): Base path for the experiment
        config (dict): Configuration dictionary
        model_name (str): Name of the model being used
    """
    config_path = os.path.join(base_path, 'training_config.json')
    
    # Create a copy of config to avoid modifying the original
    config_copy = config.copy()
    
    # Add model-specific information
    config_copy.update({
        'model_name': model_name,
        'model_description': MODEL_CONFIGS[model_name]['description']
    })
    
    # Save to file
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=4, sort_keys=True)
    
    print(f"Training configuration saved to: {config_path}")
    return config_path

def train_leave_one_subject_out(
    filenames, 
    load_data_func, 
    preprocess_data_func, 
    sliding_window_func, 
    train_model_func,
    base_path=None,
    mode=1,
    sampling_rate=50,
    window_size=None,
    overlap_size=None,
    num_classes=2
):
    """
    Perform Leave-One-Subject-Out (LOSO) cross-validation with progress tracking.
    
    Args:
        filenames (list): List of dataset file paths
        load_data_func (callable): Function to load data
        preprocess_data_func (callable): Function to preprocess data
        sliding_window_func (callable): Function to create sliding windows
        model_train_func (callable): Function to train and evaluate model
        base_path (str, optional): Base path for progress tracking
        mode (int, optional): Classification mode
        sampling_rate (int, optional): Sampling rate
        window_size (int, optional): Window size
        overlap_size (int, optional): Overlap size
        num_classes (int, optional): Number of classes
    
    Returns:
        dict: Dictionary of results for each leave-out iteration
    """
    if not base_path:
        raise ValueError("base_path must be provided for progress tracking")
        
    # Load progress if it exists
    leave_out, dict_info_names, confusion_matrices = carregar_progresso(base_path)
    model = None  # Model will be created for each fold
    print(f"Resuming from subject {leave_out} of {len(filenames)}")
    
    # Initialize or load results
    results = dict_info_names.get('results', {})
    
    try:
        while leave_out < len(filenames):
            filename = filenames[leave_out]
            print(f'Processing file {leave_out + 1}/{len(filenames)}: {filename}')
            
            # Create a copy of filenames and remove the current test file
            selected_files = filenames.copy()
            selected_files.pop(leave_out)
            
            # Load data
            train_x, train_y, test_x, test_y = load_data_func(selected_files, filename, mode)
            print(f"After load_data - train_x shape: {train_x.shape}, test_x shape: {test_x.shape}")
            
            # Create fold directory and prepare scaler path
            fold_dir = os.path.join(base_path, 'models', f'fold_{leave_out + 1}')
            os.makedirs(fold_dir, exist_ok=True)
            scaler_path = os.path.join(fold_dir, 'scaler.save')
            
            # Preprocess data and save scaler
            train_x_normalized, train_y_downsampled, \
            test_x_normalized, test_y_downsampled = preprocess_data_func(
                train_x, train_y, test_x, test_y, sampling_rate, scaler_path
            )
            print(f"After preprocess - train_x shape: {train_x_normalized.shape}, test_x shape: {test_x_normalized.shape}")
            
            # Apply sliding window
            window = window_size or 200  # Default window size of 200
            stride = window - (overlap_size or 0)  # Calculate stride based on overlap
            
            train_x, train_y = sliding_window_func(
                train_x_normalized, train_y_downsampled, 
                window, stride
            )
            print(f"After sliding window - train_x shape: {train_x.shape}")
            
            test_x, test_y = sliding_window_func(
                test_x_normalized, test_y_downsampled, 
                window, stride
            )
            
            # Set up fold info
            fold_info = {
                'fold_number': leave_out + 1,
                'test_subject': os.path.basename(filenames[leave_out]),
                'total_folds': len(filenames),
                'base_path': base_path  # Adicionando base_path ao fold_info
            }
            
            # Get model configuration and create model
            model_config = MODEL_CONFIGS[MODEL_NAME]
            input_shape = (train_x.shape[1], train_x.shape[2])
            
            # Get training config
            from config import config
            
            # Save training configuration
            if base_path:
                save_training_config(base_path, config, MODEL_NAME)
            
            # Create model with config
            model = model_config['create_fn'](
                input_shape=input_shape,
                num_classes=num_classes,
                config=config
            )
            print(f"Using {model_config['name']}: {model_config['description']}")
            print(f"Optimizer: {config.get('optimizer', 'adam')}, Learning Rate: {config.get('learning_rate', 0.001)}")
            print(f"Batch size: {config.get('batch_size', 32)}")
            print(f"Epochs: {config.get('num_epochs', 50)}")
            print(f"Early stopping patience: {config.get('patience', 10)}")
            
            # Train the model for this fold
            from config import config
            model_results = train_model_func(
                model,
                train_x,
                train_y,
                test_x,
                test_y,
                epochs=config.get('num_epochs', 50),  # Use config value, default to 50
                batch_size=config.get('batch_size', 32),  # Use config value, default to 32
                fold_info=fold_info
            )
            
            # Save progress
            salvar_progresso(
                leave_out,
                dict_info_names,
                confusion_matrices,
                model,
                base_path,
                model_results.get('history', None)
            )
            
            # Save confusion matrix as a figure
            plt.figure()
            sns.heatmap(model_results['confusion_matrix'], annot=True, fmt='d')
            plt.title(f'Confusion Matrix - Fold {leave_out}')
            results_dir = os.path.join(base_path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f'confusion_matrix_fold_{leave_out}.png'))
            plt.close()
            
            # Store results and save progress
            results[leave_out] = model_results
            confusion_matrices.append(model_results['confusion_matrix'])
            dict_info_names['results'] = results
            dict_info_names['last_completed'] = leave_out
            dict_info_names['total_subjects'] = len(filenames)
            
            # Model is already saved by ModelCheckpoint in model_trainer.py
            # We'll just verify the best model exists
            model_save_path = os.path.join(base_path, 'models', f'fold_{leave_out + 1}', 'best_model.h5')
            if os.path.exists(model_save_path):
                print(f"Best model saved to {model_save_path}")
            else:
                print(f"Warning: No best model found for fold {leave_out + 1}")
            
            # Save progress after each successful fold
            salvar_progresso(leave_out + 1, dict_info_names, confusion_matrices, model, base_path, model_results.get('history', None))
            print(f"Progress saved for fold {leave_out + 1}")
            
            # Increment leave_out for next iteration
            leave_out += 1
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Save progress even if there's an error
        salvar_progresso(leave_out, dict_info_names, confusion_matrices, model, base_path, model_results.get('history', None) if 'model_results' in locals() else None)
        raise e
    
    return results
