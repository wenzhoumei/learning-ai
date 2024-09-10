from utils.base import DataLoader

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