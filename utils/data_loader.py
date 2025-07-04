import numpy as np
import pandas as pd
import os

def one_hot_encode(labels):

    one_hot = np.zeros((labels.size, 10))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def load_data():

    # Absolute paths to the dataset files
    train_path = r"C:\Users\zakir\OneDrive\Desktop\ML\NueralNetWorkFromScratch\data\mnist_train.csv"
    test_path = r"C:\Users\zakir\OneDrive\Desktop\ML\NueralNetWorkFromScratch\data\mnist_test.csv"

    # Debug prints to verify paths
    print("[DEBUG] Train path:", train_path)
    print("[DEBUG] Exists?", os.path.exists(train_path))
    print("[DEBUG] Test path:", test_path)
    print("[DEBUG] Exists?", os.path.exists(test_path))

    # Check file existence
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "MNIST CSV files not found. Please ensure 'mnist_train.csv' and 'mnist_test.csv' are located in the correct path."
        )

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split into features and labels
    X_train = train_df.iloc[:, 1:].values / 255.0
    y_train = one_hot_encode(train_df.iloc[:, 0].values)

    X_test = test_df.iloc[:, 1:].values / 255.0
    y_test = one_hot_encode(test_df.iloc[:, 0].values)

    return X_train, y_train, X_test, y_test
