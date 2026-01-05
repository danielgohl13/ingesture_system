import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from utils import carregar_progresso, salvar_progresso
from model_configs import MODEL_CONFIGS
from config import MODEL_TYPE, MODEL_NAME, CLASSIC_MODEL_NAME, config
from feature_extractor import extract_features
from classic_model_trainer import train_classic_model

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

def save_training_config(base_path, config, model_name, model_type):
    """Save training configuration to a file.
    
    Args:
        base_path (str): Base path for the experiment
        config (dict): Configuration dictionary
        model_name (str): Name of the model being used
        model_type (str): Type of the model ('dl' or 'classic')
    """
    config_path = os.path.join(base_path, 'training_config.json')
    
    config_copy = config.copy()
    
    config_copy['model_type'] = model_type
    if model_type == 'dl':
        config_copy.update({
            'model_name': model_name,
            'model_description': MODEL_CONFIGS[model_name]['description']
        })
    else:
        config_copy['model_name'] = model_name

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
        filenames (list): List of file paths to process
        load_data_func (function): Function to load training and test data
        preprocess_data_func (function): Function to preprocess the data
        sliding_window_func (function): Function to apply sliding window
        train_model_func (function): Function to train the model
        base_path (str): Base path for saving results and progress
        mode (int): Mode for data loading (default: 1)
        sampling_rate (int): Sampling rate for preprocessing (default: 50)
        window_size (int): Size of the sliding window (default: 200)
        overlap_size (int): Overlap size for sliding window (default: 0)
        num_classes (int): Number of classes for the model (default: 2)
    """
    if not base_path:
        raise ValueError("base_path must be provided for progress tracking")
        
    leave_out, dict_info_names, confusion_matrices = carregar_progresso(base_path)
    model = None
    print(f"Resuming from subject {leave_out} of {len(filenames)}")
    
    results = dict_info_names.get('results', {})
    dict_info_names['total_subjects'] = len(filenames)
    
    try:
        while leave_out < len(filenames):
            filename = filenames[leave_out]
            print(f'Processing file {leave_out + 1}/{len(filenames)}: {filename}')
            
            selected_files = filenames.copy()
            selected_files.pop(leave_out)
            
            train_x, train_y, test_x, test_y = load_data_func(selected_files, filename, mode)
            
            fold_dir = os.path.join(base_path, 'models', f'fold_{leave_out + 1}')
            os.makedirs(fold_dir, exist_ok=True)
            scaler_path = os.path.join(fold_dir, 'scaler.save')
            
            train_x_normalized, train_y_downsampled, \
            test_x_normalized, test_y_downsampled = preprocess_data_func(
                train_x, train_y, test_x, test_y, sampling_rate, scaler_path
            )
            
            window = window_size or 200
            stride = window - (overlap_size or 0)
            
            train_x, train_y = sliding_window_func(
                train_x_normalized, train_y_downsampled, 
                window, stride
            )
            
            test_x, test_y = sliding_window_func(
                test_x_normalized, test_y_downsampled, 
                window, stride
            )
            
            fold_info = {
                'fold_number': leave_out + 1,
                'test_subject': os.path.basename(filenames[leave_out]),
                'total_folds': len(filenames),
                'base_path': base_path
            }
            
            model_results = {}

            if MODEL_TYPE == 'classic':
                print("--- Running Classic Model Pipeline ---")
                # 1. Extract features
                print("Extracting features for classic model...")
                train_x_features = extract_features(train_x)
                test_x_features = extract_features(test_x)
                print(f"Feature extraction complete. Train shape: {train_x_features.shape}, Test shape: {test_x_features.shape}")

                # 2. Train classic model
                model_results = train_classic_model(
                    train_x_features,
                    train_y,
                    test_x_features,
                    test_y,
                    fold_info=fold_info
                )
                save_training_config(base_path, config, CLASSIC_MODEL_NAME, MODEL_TYPE)

            else: # Deep Learning Pipeline
                print("--- Running Deep Learning Pipeline ---")
                model_config = MODEL_CONFIGS[MODEL_NAME]
                input_shape = (train_x.shape[1], train_x.shape[2])
                
                save_training_config(base_path, config, MODEL_NAME, MODEL_TYPE)
                
                model = model_config['create_fn'](
                    input_shape=input_shape,
                    num_classes=num_classes,
                    config=config
                )
                print(f"Using {model_config['name']}: {model_config['description']}")
                
                model_results = train_model_func(
                    model,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    epochs=config.get('num_epochs', 50),
                    batch_size=config.get('batch_size', 32),
                    fold_info=fold_info
                )
            
            # Save progress and results
            results[leave_out] = model_results
            confusion_matrices.append(model_results['confusion_matrix'])
            dict_info_names['results'] = results
            dict_info_names['last_completed'] = leave_out
            
            salvar_progresso(leave_out + 1, dict_info_names, confusion_matrices, model, base_path, model_results.get('training_history', None))
            print(f"Progress saved for fold {leave_out + 1}")
            
            leave_out += 1
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        history = model_results.get('training_history', None) if 'model_results' in locals() else None
        salvar_progresso(leave_out, dict_info_names, confusion_matrices, model, base_path, history)
        raise e
    
    return results