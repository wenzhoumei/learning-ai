from utils.base import DataLoader
from typing import List

from tensorflow.keras.datasets import mnist
import numpy as np

class MNIST(DataLoader):
    def load(self) -> np.ndarray:
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize the dataset
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # Flatten the images
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        # One-hot encode the labels
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        
        return x_train, y_train, x_test, y_test

class MNISTMiniBatch(DataLoader):
    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size

    def load(self):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the dataset
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Flatten the images
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # One-hot encode the labels
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

        # Create mini-batches for x_train and y_train
        x_train_batches, y_train_batches = self.create_batches(x_train, y_train)

        # Return the mini-batches for training and the full test dataset
        return x_train_batches, y_train_batches, x_test, y_test

    def create_batches(self, x, y):
        # Create mini-batches from the training data
        num_samples = x.shape[0]
        mini_batches_x = []
        mini_batches_y = []

        for i in range(0, num_samples, self.batch_size):
            x_batch = x[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            mini_batches_x.append(x_batch)
            mini_batches_y.append(y_batch)

        return np.array(mini_batches_x), np.array(mini_batches_y)