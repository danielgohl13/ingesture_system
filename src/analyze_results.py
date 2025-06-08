import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import MANUAL_EXPERIMENT_NAME

def load_experiment_data(experiment_name):
    """Load experiment data from progress file."""
    progress_file = os.path.join('experiments', experiment_name, 'progress', 'progresso.pkl')
    with open(progress_file, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, tuple):
            _, dict_info_names, confusion_matrices = data
            history = None
        else:
            dict_info_names = data['dict_info_names']
            confusion_matrices = data['confusion_matrices']
            history = data.get('history', None)
    return dict_info_names, confusion_matrices, history

def calculate_metrics(confusion_matrices):
    """Calculate metrics for each fold and overall."""
    if not confusion_matrices:
        raise ValueError("No confusion matrices provided")
        
    n_folds = len(confusion_matrices)
    n_classes = confusion_matrices[0].shape[0] if hasattr(confusion_matrices[0], 'shape') else len(confusion_matrices[0])
    
    # Initialize metrics
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Initialize per-class metrics
    class_metrics = {
        f'class_{i}': {
            'precision': [],
            'recall': [],
            'f1': [],
            'support': []
        } for i in range(n_classes)
    }
    
    # Calculate metrics for each fold
    for cm in confusion_matrices:
        # Convert to numpy array if it's a list
        cm_array = np.array(cm) if not isinstance(cm, np.ndarray) else cm
        
        y_true = []
        y_pred = []
        for i in range(cm_array.shape[0]):
            for j in range(cm_array.shape[1]):
                y_true.extend([i] * int(cm_array[i, j]))
                y_pred.extend([j] * int(cm_array[i, j]))
        
        if not y_true:  # Skip if no samples in this fold
            continue
        
        # Calculate overall metrics
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Calculate per-class metrics
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(n_classes):
            class_metrics[f'class_{i}']['precision'].append(precision[i] if i < len(precision) else 0)
            class_metrics[f'class_{i}']['recall'].append(recall[i] if i < len(recall) else 0)
            class_metrics[f'class_{i}']['f1'].append(f1[i] if i < len(f1) else 0)
            class_metrics[f'class_{i}']['support'].append(np.sum(np.array(y_true) == i))
    
    # Calculate mean and std for overall metrics
    summary = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in metrics.items()
    }
    
    # Calculate mean and std for per-class metrics
    class_summary = {}
    for class_name, metrics_dict in class_metrics.items():
        class_summary[class_name] = {
            'precision': {
                'mean': np.mean(metrics_dict['precision']),
                'std': np.std(metrics_dict['precision'])
            },
            'recall': {
                'mean': np.mean(metrics_dict['recall']),
                'std': np.std(metrics_dict['recall'])
            },
            'f1': {
                'mean': np.mean(metrics_dict['f1']),
                'std': np.std(metrics_dict['f1'])
            },
            'support': int(np.mean(metrics_dict['support']))
        }
    
    return metrics, summary, class_summary

def plot_metrics(metrics, save_path):
    """Plot metrics across folds."""
    plt.figure(figsize=(12, 6))
    x = range(1, len(metrics['accuracy']) + 1)
    
    for metric in metrics:
        plt.plot(x, metrics[metric], marker='o', label=metric.capitalize())
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Metrics Across Folds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'metrics_across_folds.png'))
    plt.close()

def plot_average_confusion_matrix(confusion_matrices, save_path):
    """Plot average confusion matrix across all folds."""
    # Convert all matrices to numpy arrays if they aren't already
    np_matrices = [np.array(cm) if not isinstance(cm, np.ndarray) else cm for cm in confusion_matrices]
    
    # Find maximum shape
    max_shape = max(cm.shape[0] for cm in np_matrices)
    padded_matrices = []
    
    for cm in np_matrices:
        if cm.shape[0] < max_shape:
            # Pad with zeros
            padded_cm = np.zeros((max_shape, max_shape))
            padded_cm[:cm.shape[0], :cm.shape[1]] = cm
            padded_matrices.append(padded_cm)
        else:
            padded_matrices.append(cm)
    
    # Compute average
    if padded_matrices:  # Check if there are any matrices to average
        avg_cm = np.mean(padded_matrices, axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Average Confusion Matrix Across All Folds')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(os.path.join(save_path, 'average_confusion_matrix.png'))
        plt.close()

def plot_learning_curves(history, results_path):
    """Plot learning curves from training history."""
    if not history:
        return

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'overall_loss.png'))
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_path, 'overall_accuracy.png'))
    plt.close()

def save_results_to_csv(metrics, summary, class_summary, save_path):
    """Save metrics to CSV files."""
    # Save per-fold metrics
    df_folds = pd.DataFrame(metrics)
    df_folds.index = [f'Fold_{i+1}' for i in range(len(df_folds))]
    df_folds.to_csv(os.path.join(save_path, 'fold_metrics.csv'))
    
    # Save summary metrics
    df_summary = pd.DataFrame({
        metric: {
            'Mean': data['mean'],
            'Std': data['std']
        }
        for metric, data in summary.items()
    }).transpose()
    df_summary.to_csv(os.path.join(save_path, 'summary_metrics.csv'))
    
    # Save per-class metrics
    class_data = []
    for class_name, metrics_dict in class_summary.items():
        row = {'Class': class_name}
        for metric, values in metrics_dict.items():
            if metric != 'support':
                row[f'{metric}_mean'] = values['mean']
                row[f'{metric}_std'] = values['std']
            else:
                row['support'] = values
        class_data.append(row)
    
    df_class = pd.DataFrame(class_data)
    df_class.to_csv(os.path.join(save_path, 'class_metrics.csv'), index=False)

def main():
    try:
        if not MANUAL_EXPERIMENT_NAME:
            raise ValueError("Por favor, defina MANUAL_EXPERIMENT_NAME no arquivo config.py com o nome do experimento que deseja analisar")
        
        print(f"Analisando resultados do experimento: {MANUAL_EXPERIMENT_NAME}")
        
        # Create results directory
        results_path = os.path.join('experiments', MANUAL_EXPERIMENT_NAME, 'results')
        os.makedirs(results_path, exist_ok=True)
        print(f"Diretório de resultados: {os.path.abspath(results_path)}")
        
        # Load experiment data
        print("Carregando dados do experimento...")
        dict_info_names, confusion_matrices, history = load_experiment_data(MANUAL_EXPERIMENT_NAME)
        
        if not confusion_matrices:
            raise ValueError("Nenhuma matriz de confusão encontrada no arquivo de progresso")
        
        print(f"Dados carregados: {len(confusion_matrices)} matrizes de confusão encontradas")
        
        # Calculate metrics
        print("Calculando métricas...")
        metrics, summary, class_summary = calculate_metrics(confusion_matrices)
        
        # Generate and save visualizations
        print("Gerando visualizações...")
        plot_metrics(metrics, results_path)
        plot_average_confusion_matrix(confusion_matrices, results_path)
        plot_learning_curves(history, results_path)
        
        # Save metrics to CSV
        print("Salvando métricas em arquivos CSV...")
        save_results_to_csv(metrics, summary, class_summary, results_path)
        
        # Print summary
        print("\n=== Resumo dos Resultados ===")
        print("\n--- Métricas Globais ---")
        for metric, values in summary.items():
            print(f"\n{metric.capitalize()}:")
            print(f"  Média: {values['mean']:.4f}")
            print(f"  Desvio Padrão: {values['std']:.4f}")
        
        # Print per-class metrics
        print("\n--- Métricas por Classe ---")
        for class_name, metrics_dict in class_summary.items():
            print(f"\n{class_name} (Amostras: {metrics_dict['support']}):")
            for metric in ['precision', 'recall', 'f1']:
                print(f"  {metric.capitalize()}: {metrics_dict[metric]['mean']:.4f} ± {metrics_dict[metric]['std']:.4f}")
            
        print("\nAnálise concluída com sucesso!")
        print(f"Resultados salvos em: {os.path.abspath(results_path)}")
        
    except Exception as e:
        print(f"\nErro durante a análise dos resultados: {str(e)}")
        print("Verifique se o experimento foi executado corretamente e se o MANUAL_EXPERIMENT_NAME está configurado corretamente.")
        raise

if __name__ == '__main__':
    main()
