from utils.base import WeightInitializer

import numpy as np

class Random(WeightInitializer):
    def initialize(self, input_size: int, output_size: int, range: float = 0.01, offset: int = 0) -> np.ndarray:
        # Random initialization with small random numbers
        return np.random.randn(input_size, output_size) * range + offset

class Xavier(WeightInitializer):
    def initialize(self, input_size: int, output_size: int) -> np.ndarray:
        # Xavier initialization
        return np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))