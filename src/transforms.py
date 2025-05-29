import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def sliding_window(x, y, window, stride):
    data, target = [], []
    start = 0
    while start + window < x.shape[0]:
        end = start + window
        x_segment = x[start:end]
        y_segment = y[start:end]

        labels_name, labels_count = np.unique(y_segment, return_counts=True)
        if len(labels_name) > 1:
            if(np.amax(labels_count) > window*0.75):
                target.append(np.argmax(np.bincount(y_segment)))
                data.append(x_segment)
        else:
            target.append(y_segment[0])
            data.append(x_segment)
        start += stride

    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.int64)

    return data, target

#Pre-processamento salvando o standart scaler
def preprocess_data(train_x, train_y, test_x, test_y, sampling_rate, scaler_save_path=None):
    def downsample(data_x, data_y, original_rate=200, new_rate=sampling_rate):
        downsample_factor = int(original_rate / new_rate)
        downsampled_data_x = data_x[::downsample_factor]
        downsampled_data_y = []
        for i in range(0, len(data_y), downsample_factor):
            y_segment = data_y[i:i+downsample_factor]
            if len(y_segment) == 0:
                break
            majority_label = np.argmax(np.bincount(y_segment))
            downsampled_data_y.append(majority_label)
        downsampled_data_y = np.array(downsampled_data_y)
        return downsampled_data_x, downsampled_data_y

    train_x_downsampled, train_y_downsampled = downsample(train_x, train_y)
    test_x_downsampled, test_y_downsampled = downsample(test_x, test_y)

    scaler = StandardScaler()
    train_x_normalized = scaler.fit_transform(train_x_downsampled)
    test_x_normalized = scaler.transform(test_x_downsampled)
    
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved to {scaler_save_path}")
    
    return train_x_normalized, train_y_downsampled, test_x_normalized, test_y_downsampled

    