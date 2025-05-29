import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler    
from config import config

cols = ['accX', 'accY', 'accZ', 'asX', 'asY', 'asZ', 'label']


def split_x_y(data):
    data_x = data.iloc[:, 0:-1]
    data_y = np.array(data.iloc[:, -1])
    return data_x, data_y

def load_data(train_files, test_files, mode=config['mode']):
    """
    Load training and test data from CSV files with optional label processing.

    Args:
        train_files (list or str): Training file path(s)
        test_files (list or str): Test file path(s)
        mode (int, optional): Label processing mode. 
            0: No modification (default)
            1: Binary classification (0 if not 1, else 1)
            2: Custom label processing (can be extended)

    Returns:
        tuple: X_train, y_train, X_test, y_test as numpy arrays
    """
    # Ensure inputs are lists
    train_files = [train_files] if isinstance(train_files, str) else train_files
    test_files = [test_files] if isinstance(test_files, str) else test_files

    # Validate inputs
    if not train_files or not test_files:
        raise ValueError("Train and test file lists cannot be empty")

    # Initialize data containers
    data_x = np.empty((0, 6))
    data_y = np.empty((0))
    test_data_x = np.empty((0, 6))
    test_data_y = np.empty((0))

    # Label processing function
    def process_labels(df, mode):
        """Process labels based on mode"""
        if mode == 1:
            # Binary classification: 0 if not 1, else 1
            df['label'] = df['label'].apply(lambda x: 0 if x != 1 else 1)
        elif mode == 2:
            # Placeholder for custom label processing
            # Add your custom label processing logic here
            pass
        return df

    # Load training data
    print('Loading dataset train files ...')
    for i, filename in enumerate(train_files, 1):
        try:
            print(f"... reading {filename}")
            df = pd.read_csv(filename, usecols=cols)
            
            # Process labels based on mode
            df = process_labels(df, mode)
            
            x, y = split_x_y(df)
            data_x = np.vstack((data_x, x.to_numpy()))
            data_y = np.concatenate([data_y, y])
            
            # Optional: print progress
            print(f"Processed file {i}/{len(train_files)}")
        
        except FileNotFoundError:
            print(f"Warning: Training file {filename} not found. Skipping.")
        except Exception as e:
            print(f"Error processing training file {filename}: {e}")
            raise

    # Load test data
    print('Loading dataset test files ...')
    for filename in test_files:
        try:
            print(f"... reading {filename}")
            df = pd.read_csv(filename, usecols=cols)
            
            # Process labels based on mode
            df = process_labels(df, mode)
            
            x, y = split_x_y(df)
            test_data_x = np.vstack((test_data_x, x.to_numpy()))
            test_data_y = np.concatenate([test_data_y, y])
        
        except FileNotFoundError:
            print(f"Warning: Test file {filename} not found. Skipping.")
        except Exception as e:
            print(f"Error processing test file {filename}: {e}")
            raise

    # Validate data
    if len(data_x) == 0 or len(test_data_x) == 0:
        raise ValueError("No data could be loaded from the provided files")

    # Convert and return data
    return (
        data_x.astype(float), 
        data_y.astype(int), 
        test_data_x.astype(float), 
        test_data_y.astype(int)
    )


def carregar_progresso(base_path):
    leave_out = 0
    dict_info_names = {}
    confusion_matrices = []

    try:
        with open(os.path.join(base_path, 'progresso_leave_out.pkl'), 'rb') as f:
            leave_out = pickle.load(f)
    except FileNotFoundError:
        print("arquivo não encontrado.")

    try:
        with open(os.path.join(base_path, 'progresso_dict_info_names.pkl'), 'rb') as f:
            dict_info_names = pickle.load(f)
    except FileNotFoundError:
        pass

    try:
        with open(os.path.join(base_path, 'progresso_confusion_matrices.pkl'), 'rb') as f:
            confusion_matrices = pickle.load(f)
    except FileNotFoundError:
        pass

    return leave_out, dict_info_names, confusion_matrices


def salvar_progresso(leave_out, dict_info_names, confusion_matrices, base_path):
    """Save progress to a file"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with open(os.path.join(base_path, 'progress', 'progresso.pkl'), 'wb') as f:
        pickle.dump((leave_out, dict_info_names, confusion_matrices), f)
