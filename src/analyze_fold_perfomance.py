import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from analyze_results import load_experiment_data
from config import MANUAL_EXPERIMENT_NAME

def calculate_fold_metrics(confusion_matrix):
    """Calculate TP, FP, FN, precision, recall and F1 for binary classification."""
    # Assume binary classification with classes 0 and 1
    # Confusion matrix format:
    # [[TN, FP],
    #  [FN, TP]]
    cm = np.array(confusion_matrix)
    
    # For binary classification, we can extract TP, FP, FN
    if cm.shape == (2, 2):
        TN, FP = cm[0]
        FN, TP = cm[1]
    else:
        # Handle multi-class by considering the last class as positive
        TP = cm[-1, -1]
        FP = cm[:-1, -1].sum()
        FN = cm[-1, :-1].sum()
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'precision': precision,
        'recall': recall,
        'f1-score': f1
    }

def generate_fold_performance_table(experiment_name):
    """Generate a table with performance metrics for each fold."""
    # Load experiment data
    try:
        dict_info_names, confusion_matrices, _ = load_experiment_data(experiment_name)
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None
    
    # Initialize results list
    results = []
    
    # Calculate metrics for each fold
    for fold_idx, cm in enumerate(confusion_matrices, 1):
        fold_metrics = calculate_fold_metrics(cm)
        fold_metrics['# of subject'] = fold_idx
        results.append(fold_metrics)
    
    # Create DataFrame
    columns = ['# of subject', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1-score']
    df = pd.DataFrame(results, columns=columns)
    
    # Reorder columns
    df = df[columns]
    
    # Calculate mean and std for metrics
    metrics = ['precision', 'recall', 'f1-score']
    stats = {
        'Mean': df[metrics].mean(),
        'Std': df[metrics].std()
    }
    stats_df = pd.DataFrame(stats).T
    
    return df, stats_df

def save_results(df, stats_df, output_dir):
    """Save results to CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fold metrics
    fold_metrics_path = os.path.join(output_dir, 'fold_performance.csv')
    df.to_csv(fold_metrics_path, index=False, float_format='%.4f')
    print(f"Fold performance saved to: {fold_metrics_path}")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'fold_performance_stats.csv')
    stats_df.to_csv(stats_path, float_format='%.4f')
    print(f"Performance statistics saved to: {stats_path}")
    
    # Print results
    print("\nFold Performance:")
    print(df.to_string(index=False))
    
    print("\nPerformance Statistics:")
    print(stats_df.to_string())

def main():
    # Get experiment name from command line or use default
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = MANUAL_EXPERIMENT_NAME
    
    # Set output directory
    output_dir = os.path.join('experiments', experiment_name, 'results')
    
    # Generate and save results
    results = generate_fold_performance_table(experiment_name)
    if results:
        df, stats_df = results
        save_results(df, stats_df, output_dir)

if __name__ == '__main__':
    main()