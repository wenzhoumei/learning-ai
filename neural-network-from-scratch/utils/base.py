from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class BiasInitializer(ABC):
    @abstractmethod
    def initialize(self, output_size: int) -> np.ndarray:
        pass

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class WeightInitializer(ABC):
    @abstractmethod
    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        pass

class DataLoader(ABC):
    @abstractmethod
    def load(self, input_size: int, output_size: int) -> np.ndarray:
        pass

class BatchSplitter(ABC):
    @abstractmethod
    def split(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        pass