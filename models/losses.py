import numpy as np

def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / n
