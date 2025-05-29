import os
import pickle
import shutil

def carregar_progresso(base_path):
    """Load progress from a file.
    
    Args:
        base_path (str): Base path where progress files are stored
        
    Returns:
        tuple: leave_out, dict_info_names, confusion_matrices
    """
    progress_file = os.path.join(base_path, 'progress', 'progresso.pkl')
    backup_file = os.path.join(base_path, 'progress', 'progresso.pkl.bak')
    
    # Try to load the main progress file
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                leave_out = progress_data['leave_out']
                dict_info_names = progress_data['dict_info_names']
                confusion_matrices = progress_data['confusion_matrices']
                history = progress_data.get('history', {})
                print(f"Loaded progress from {progress_file}")
                print(f"Last completed fold: {dict_info_names.get('last_completed', -1)}")
                return leave_out, dict_info_names, confusion_matrices
        except Exception as e:
            print(f"Error loading progress file: {str(e)}")
            if os.path.exists(backup_file):
                print("Attempting to load backup file...")
                with open(backup_file, 'rb') as f:
                    leave_out, dict_info_names, confusion_matrices = pickle.load(f)
                    print(f"Loaded progress from backup file")
                    return leave_out, dict_info_names, confusion_matrices
    
    # If no progress file exists or both files are corrupted, start fresh
    print("No previous progress found. Starting from beginning.")
    leave_out = 0
    dict_info_names = {
        'results': {},
        'last_completed': -1,
        'total_subjects': None
    }
    confusion_matrices = []
    
    return leave_out, dict_info_names, confusion_matrices

def salvar_progresso(leave_out, dict_info_names, confusion_matrices, model, base_path, history=None):
    """Save progress to a file with backup.
    
    Args:
        leave_out (int): Current leave-out iteration
        dict_info_names (dict): Dictionary of information about subjects
        confusion_matrices (list): List of confusion matrices
        model: The trained model for this fold (not saved in pickle)
        base_path (str): Base path to save progress files
        history (dict, optional): Training history containing loss and metrics
    """
    progress_file = os.path.join(base_path, 'progress', 'progresso.pkl')
    backup_file = os.path.join(base_path, 'progress', 'progresso.pkl.bak')
    
    # First create a backup of the existing progress file if it exists
    if os.path.exists(progress_file):
        try:
            shutil.copy2(progress_file, backup_file)
        except Exception as e:
            print(f"Warning: Could not create backup file: {str(e)}")
    
    # Save the new progress (we don't save the model in pickle)
    try:
        with open(progress_file, 'wb') as f:
            pickle.dump((leave_out, dict_info_names, confusion_matrices), f)
        print(f"Progress saved to {progress_file}")
    except Exception as e:
        print(f"Error saving progress: {str(e)}")
        raise e
