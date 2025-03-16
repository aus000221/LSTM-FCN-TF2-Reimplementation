import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# written by yc4324 & kl3606
def load_ucr_dataset(dataset_name, ucr_dir):
    """
    Load a UCR dataset by name.
    assume all input data is univariate time series
    Args:
    - dataset_name: Name of the dataset (e.g., "ACSF1").
    - ucr_dir: Directory containing UCR datasets.

    Returns:
    - X_train, y_train, X_test, y_test: Processed train/test data and labels.
    """
    train_file = os.path.join(ucr_dir, f"{dataset_name}_TRAIN.csv")
    test_file = os.path.join(ucr_dir, f"{dataset_name}_TEST.csv")

    # Load CSV files
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    #fill missing value with 0
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)
    
    # First column is the label, rest are time series data
    y_train = train_data.iloc[:, 0]
    
    X_train = train_data.iloc[:, 1:]
    X_train = X_train.to_numpy()
    
    num_classes = len(np.unique(y_train))
    
    # normalize y to number: (0,number of classes-1) interger
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (num_classes - 1) #normalize y t
    X_train = train_data.iloc[:, 1:]
    X_train = X_train.to_numpy()
    
    # the same for test dataset
    y_test = test_data.iloc[:, 0]
    X_test = test_data.iloc[:, 1:]
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (num_classes - 1)
    X_test = X_test.to_numpy()

    # Normalize x 
    X_train = (X_train - np.mean(X_train, axis=-1, keepdims=True)) / (np.std(X_train, axis=-1, keepdims=True) + 1e-8)
    X_test = (X_test - np.mean(X_test, axis=-1, keepdims=True)) / (np.std(X_test, axis=-1, keepdims=True) + 1e-8)

    # Reshape for LSTM input (add feature dimension)
    X_train =  X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    # One-hot encode labels
    y_train_ohp = to_categorical(y_train, num_classes=num_classes)
    y_test_ohp = to_categorical(y_test, num_classes=num_classes)

    return X_train, y_train_ohp, X_test, y_test_ohp, y_train, y_test, num_classes
