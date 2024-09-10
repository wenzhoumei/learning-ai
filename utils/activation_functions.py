
from utils.base import ActivationFunction

import numpy as np

class ReLu(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.activate(x)
        return sig * (1 - sig)

class Tanh(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2

class LeakyReLu(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

class Softmax(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.activate(x)
        return s * (1 - s)

class Swish(ActivationFunction):
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def activate(self, x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-self.beta * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = 1 / (1 + np.exp(-self.beta * x))
        return sigmoid + x * self.beta * sigmoid * (1 - sigmoid)

class Softplus(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class ELU(ActivationFunction):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def activate(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.activate(x) + self.alpha)

class SELU(ActivationFunction):
    def __init__(self):
        self.alpha = 1.67326
        self.scale = 1.0507

    def activate(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.where(x > 0, 1, self.activate(x) + self.alpha)

class GELU(ActivationFunction):
    def activate(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + \
               (0.5 * x * (1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))**2) * \
                (np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(x, 2))))