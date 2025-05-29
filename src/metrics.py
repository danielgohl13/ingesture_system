import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    """
    F1-Score metric for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        F1-Score
    """
    # Convert probabilities to binary predictions
    y_pred = K.round(y_pred)
    
    # Calculate true positives, false positives and false negatives
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return K.mean(f1)
