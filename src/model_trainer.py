import datetime
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    classification_report,
    accuracy_score
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Função auxiliar para converter tipos numpy para tipos nativos do Python
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=32, fold_info=None):
    """Train a model and track its performance.
    
    Args:
        model (tf.keras.Model): The compiled model to train
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray, optional): Validation data
        y_val (numpy.ndarray, optional): Validation labels
        epochs (int, optional): Number of training epochs. If None, uses config value.
        batch_size (int): Batch size for training
        fold_info (dict, optional): Information about the current fold
        
    Returns:
        dict: Training results including accuracy, loss and confusion matrix
    """
    # Import config to get number of epochs
    from config import config
    
    # Use configured epochs if not explicitly specified
    if epochs is None:
        epochs = config.get('num_epochs', 10)  # Default to 10 if not specified
    # Convert labels to one-hot encoding if needed for multiclass classification
    from tensorflow.keras.utils import to_categorical
    
    # Store original shapes for later use
    y_train_original_shape = y_train.shape
    y_val_original_shape = y_val.shape if y_val is not None else None
    
    # Check if we need to convert to one-hot encoding
    num_classes = model.output_shape[-1] if len(model.output_shape) > 1 else 2
    
    # Convert training labels
    if len(y_train.shape) == 1 or (len(y_train.shape) > 1 and y_train.shape[1] == 1):
        # Convert to one-hot if we have more than 2 classes or if it's binary but needs to be one-hot
        if num_classes > 2 or (num_classes == 2 and len(y_train.shape) == 1):
            y_train = to_categorical(y_train, num_classes=num_classes)
    
    # Convert validation labels if they exist
    if y_val is not None:
        if len(y_val.shape) == 1 or (len(y_val.shape) > 1 and y_val.shape[1] == 1):
            if num_classes > 2 or (num_classes == 2 and len(y_val.shape) == 1):
                y_val = to_categorical(y_val, num_classes=num_classes)
    
    # Prepare callbacks for early stopping and model checkpointing
    if fold_info and 'fold_number' in fold_info:
        from config import get_experiment_path
        base_path = get_experiment_path()
        results_dir = os.path.join(base_path, 'models', f'fold_{fold_info["fold_number"]}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Custom F1 score metric callback
        class F1ScoreCallback(tf.keras.callbacks.Callback):
            def __init__(self, val_data, val_labels):
                super().__init__()
                self.val_data = val_data
                self.val_labels = val_labels
                self.best_f1 = 0
            
            def on_epoch_end(self, epoch, logs=None):
                predictions = self.model.predict(self.val_data)
                
                # Handle both binary and multiclass predictions
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predicted_classes = np.argmax(predictions, axis=1)
                else:
                    predicted_classes = (predictions > 0.5).astype(int).flatten()
                
                # Handle both one-hot encoded and integer labels
                if len(self.val_labels.shape) > 1 and self.val_labels.shape[1] > 1:
                    true_classes = np.argmax(self.val_labels, axis=1)
                else:
                    true_classes = self.val_labels.flatten()
                
                # Calculate weighted F1 score for multiclass, binary for binary
                average = 'binary' if len(np.unique(true_classes)) <= 2 else 'weighted'
                f1 = f1_score(true_classes, predicted_classes, average=average)
                
                logs['val_f1_score'] = f1
                
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    print(f'\nImproved F1 Score: {f1:.4f}')
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=config.get('patience', 10),  # Use patience from config, default to 10
            restore_best_weights=True,
            verbose=1
        )
        
        # Create results directory if it doesn't exist
        results_dir_path = Path(results_dir)
        results_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Criar diretório SavedModel
        saved_model_dir = results_dir_path / 'saved_model'
        saved_model_dir.mkdir(exist_ok=True)
        
        # Callback para salvar o melhor modelo no formato SavedModel (.pb)
        model_checkpoint = ModelCheckpoint(
            filepath=str(saved_model_dir / 'best_model'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            save_format='tf',  # Isso garante o formato SavedModel (.pb)
            verbose=1
        )
        
        # Callback para salvar os pesos em formato .h5 (opcional)
        weights_path = results_dir_path / 'best_weights.h5'
        checkpoint_weights = ModelCheckpoint(
            filepath=str(weights_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        )
        
        # F1 Score tracking callback
        f1_callback = F1ScoreCallback(X_val, y_val) if X_val is not None else None
        
        callbacks = [early_stopping, model_checkpoint, checkpoint_weights]
        if f1_callback:
            callbacks.append(f1_callback)
    else:
        callbacks = None
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )
    
    # Get predictions and generate classification report
    val_data = X_val if X_val is not None else X_train
    val_labels = y_val if y_val is not None else y_train
    
    # Generate predictions
    y_pred = model.predict(val_data, verbose=0)
    
    # Get predicted class indices
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        # For binary classification with sigmoid output
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    
    # Convert one-hot encoded labels back to class indices if needed
    if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
        val_labels = np.argmax(val_labels, axis=1)
    # Ensure val_labels is 1D
    val_labels = val_labels.flatten()
    
    # Ensure y_pred_classes is 1D
    y_pred_classes = y_pred_classes.flatten()
    
    # Generate classification report
    report = classification_report(
        val_labels, 
        y_pred_classes,
        output_dict=True
    )
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, y_pred_classes)
    f1 = f1_score(val_labels, y_pred_classes, average='weighted')
    cm = confusion_matrix(val_labels, y_pred_classes)
    
    # Get training history metrics
    train_metrics = {}
    if hasattr(history, 'history'):
        train_metrics = {
            'train_loss': history.history.get('loss', []),
            'train_accuracy': history.history.get('accuracy', []),
            'train_f1': history.history.get('f1_score', []),
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', []),
            'val_f1': history.history.get('val_f1_score', [])
        }
    
    # Create results dictionary with all metrics
    results = {
        'metrics': {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'loss': history.history['val_loss'][-1] if hasattr(history, 'history') and 'val_loss' in history.history else None,
            'best_epoch': np.argmin(history.history['val_loss']) + 1 if hasattr(history, 'history') and 'val_loss' in history.history else None
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'training_history': train_metrics,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Save reports and plots if fold info is available
    if fold_info and 'fold_number' in fold_info:
        # Ensure base_path is set, default to current directory if not
        base_path = fold_info.get('base_path', '.')
        report_path = Path(base_path) / 'results' / f'fold_{fold_info["fold_number"]}'
        report_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        with open(report_path / 'training_results.json', 'w') as f:
            # Converter todos os valores numpy para tipos nativos do Python
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        
        # Save classification report as text and JSON
        report_text = classification_report(val_labels, y_pred_classes)
        report_dict = classification_report(val_labels, y_pred_classes, output_dict=True)
        
        with open(report_path / 'classification_report.txt', 'w') as f:
            f.write(report_text)
            
        with open(report_path / 'classification_report.json', 'w') as f:
            json.dump(convert_to_serializable(report_dict), f, indent=4, ensure_ascii=False)
            
        # Save metrics summary
        metrics_summary = {
            'best_epoch': results['metrics']['best_epoch'],
            'final_accuracy': results['metrics']['accuracy'],
            'final_f1_score': results['metrics']['f1_score'],
            'final_loss': results['metrics']['loss'],
            'best_val_loss': min(results['training_history']['val_loss']) if 'val_loss' in results['training_history'] else None,
            'best_val_accuracy': max(results['training_history']['val_accuracy']) if 'val_accuracy' in results['training_history'] else None,
            'training_completed': True,
            'last_epoch': len(results['training_history'].get('loss', []))
        }
        with open(report_path / 'metrics_summary.json', 'w') as f:
            # Converter para tipos nativos antes de serializar
            serializable_metrics = convert_to_serializable(metrics_summary)
            json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)
        
        # Plot confusion matrix
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(report_path / 'confusion_matrix.png')
            plt.close()
            
            # Training History Plots
            if hasattr(history, 'history'):
                history_dict = history.history
                
                # Plot training & validation accuracy values
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history_dict['accuracy'])
                plt.plot(history_dict['val_accuracy'])
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='lower right')
                
                # Plot training & validation loss values
                plt.subplot(1, 2, 2)
                plt.plot(history_dict['loss'])
                plt.plot(history_dict['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper right')
                
                plt.tight_layout()
                plt.savefig(report_path / 'training_history.png')
                plt.close()
                
                # Save F1 score plot if available
                if 'f1_score' in history_dict and 'val_f1_score' in history_dict:
                    plt.figure(figsize=(6, 4))
                    plt.plot(history_dict['f1_score'])
                    plt.plot(history_dict['val_f1_score'])
                    plt.title('Model F1 Score')
                    plt.ylabel('F1 Score')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Validation'], loc='lower right')
                    plt.tight_layout()
                    plt.savefig(report_path / 'f1_score_history.png')
                    plt.close()
            
            print(f"Saved evaluation results to: {report_path}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    return results
