
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocessing.more_preprocessing.process2 import iden_tensor

def generate_dummy_data(num_samples=1000, input_size=10, num_classes=2):
    """
    Generate synthetic classification data.
    """
    X = np.random.randn(num_samples, input_size)
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y

def preprocess_data(X, y, test_size=0.2, normalize=True):
    """
    Normalize features and split data into train/test sets.
    """
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    X_train = iden_tensor(X_train)
    return X_train, X_test, y_train, y_test