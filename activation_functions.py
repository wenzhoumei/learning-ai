# Import necessary modules
from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

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