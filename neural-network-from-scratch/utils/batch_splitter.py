from utils.base import BatchSplitter

import numpy as np
from typing import List, Tuple


class MiniBatch(BatchSplitter):
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def split(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        m = x.shape[0]
        permutation = np.random.permutation(m)
        x_shuffled = x[permutation]
        y_shuffled = y[permutation]

        mini_batches = []
        for i in range(0, m, self.batch_size):
            x_mini_batch = x_shuffled[i:i + self.batch_size]
            y_mini_batch = y_shuffled[i:i + self.batch_size]
            mini_batches.append((x_mini_batch, y_mini_batch))

        return mini_batches

class FullBatch(BatchSplitter): # Basically MiniBatch with batch_size of len(x)
    def split(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(x, y)]

class Stochastic(BatchSplitter): # Basically MiniBatch with batch_size of 1
    def split(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        m = x.shape[0]
        stochastic_examples = [(x[i:i+1], y[i:i+1]) for i in range(m)]
        return stochastic_examples