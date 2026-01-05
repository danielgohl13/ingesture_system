import datetime
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from config import CLASSIC_MODEL_NAME, classic_model_config, NUM_CLASSES
from model_trainer import convert_to_serializable

def get_classic_model():
    """
    Instantiates a classic machine learning model based on the configuration.
    """
    model_name = CLASSIC_MODEL_NAME
    params = classic_model_config.get(model_name, {})
    
    model_constructors = {
        'RandomForest': RandomForestClassifier,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
        'AdaBoost': AdaBoostClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'NaiveBayes': GaussianNB,
    }
    
    if model_name not in model_constructors:
        raise ValueError(f"Model {model_name} is not supported. Choose from {list(model_constructors.keys())}")
    
    model = model_constructors[model_name](**params)
    
    print(f"Using classic model: {model_name} with params: {params}")
    return model

def train_classic_model(X_train, y_train, X_val, y_val, fold_info=None):
    """
    Trains and evaluates a classic machine learning model.
    """
    # 1. Preprocessing: Impute NaNs and scale features
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 2. Get and train the model
    model = get_classic_model()
    model.fit(X_train, y_train)

    # 3. Evaluate the model, ensuring all classes are considered
    y_pred = model.predict(X_val)
    
    # Define labels to ensure consistency across all folds
    labels = list(range(NUM_CLASSES))
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted', labels=labels)
    report = classification_report(y_val, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    results = {
        'metrics': {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.datetime.now().isoformat()
    }

    # 4. Save results and plots
    if fold_info and 'fold_number' in fold_info:
        base_path = fold_info.get('base_path', '.')
        report_path = Path(base_path) / 'results' / f'fold_{fold_info["fold_number"]}'
        report_path.mkdir(parents=True, exist_ok=True)

        with open(report_path / 'training_results.json', 'w') as f:
            serializable_results = convert_to_serializable(results)
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)

        # Ensure the text report also considers all labels
        report_text = classification_report(y_val, y_pred, labels=labels, zero_division=0)
        with open(report_path / 'classification_report.txt', 'w') as f:
            f.write(report_text)
            
        plt.figure(figsize=(10, 8))
        # Add labels to the heatmap for correct axis ticks
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(report_path / 'confusion_matrix.png')
        plt.close()
        
        print(f"Saved classic model evaluation results to: {report_path}")

    return results