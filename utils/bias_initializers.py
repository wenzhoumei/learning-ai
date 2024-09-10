from utils.base import BiasInitializer

import numpy as np

class Zero(BiasInitializer):
    def initialize(self, output_size: int) -> np.ndarray:
        # Initialize biases to zero
        return np.zeros((1, output_size))

class Random(BiasInitializer):
    def initialize(self, output_size: int) -> np.ndarray:
        # Initialize biases to small random values
        return np.random.randn(1, output_size) * 0.01