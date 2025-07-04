import numpy as np
from models.activations import relu, relu_derivative, softmax
from models.losses import cross_entropy_loss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier Initialization
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.output = softmax(self.z2)
        return self.output

    def compute_loss(self, y_pred, y_true):
        return cross_entropy_loss(y_pred, y_true)

    def backward(self, X, y_true, y_pred, lr):
        n = X.shape[0]

        dz2 = (y_pred - y_true) / n
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.weights2 -= lr * dw2
        self.bias2 -= lr * db2
        self.weights1 -= lr * dw1
        self.bias1 -= lr * db1
