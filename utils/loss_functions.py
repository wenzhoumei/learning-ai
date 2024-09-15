from utils.base import LossFunction

import numpy as np

class MSE(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class CrossEntropy(LossFunction):
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon # Small constant to prevent log(0)

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1. - self.epsilon)
        return -y_true / y_pred