# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SmallNet
from preprocessing.process import generate_dummy_data, preprocess_data

def main():
    X, y = generate_dummy_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = SmallNet()
    torch.save(model, "smallnet.ckpt")
    print("Model saved.")

if __name__ == "__main__":
    main()
