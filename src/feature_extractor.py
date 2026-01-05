import numpy as np
from scipy.stats import kurtosis, skew

def extract_features(segments):
    """
    Extracts statistical features from a list of segments.

    For each segment, it calculates:
    - Mean
    - Standard Deviation
    - Variance
    - Max value
    - Min value
    - Range (Max - Min)
    - Kurtosis
    - Skewness
    
    Args:
        segments (np.ndarray): A 3D numpy array of shape (num_segments, window_size, num_channels).

    Returns:
        np.ndarray: A 2D numpy array of shape (num_segments, num_features).
    """
    features = []
    for segment in segments:
        segment_features = []
        # segment shape is (window_size, num_channels)
        for i in range(segment.shape[1]):
            signal = segment[:, i]
            
            std_dev = np.std(signal)
            
            # Handle cases with near-zero variance to avoid precision loss warnings
            if std_dev < 1e-9:
                kurt = 0
                sk = 0
            else:
                kurt = kurtosis(signal)
                sk = skew(signal)

            # Basic statistical features
            segment_features.extend([
                np.mean(signal),
                std_dev,
                np.var(signal),
                np.max(signal),
                np.min(signal),
                np.max(signal) - np.min(signal),  # Range
                kurt,
                sk
            ])
        features.append(segment_features)
    
    return np.array(features)