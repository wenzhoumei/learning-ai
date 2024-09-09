import numpy as np
from tensorflow.keras.datasets import mnist
from typing import List
from activation_functions import ReLu, Softmax, ActivationFunction

def loadData():
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

# Define the neural network class
class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: List[ActivationFunction]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.num_layers - 1)]
        self.activations = activations

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations_cache = [x]
        self.z_values = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations_cache[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activations[i].activate(z)
            self.activations_cache.append(a)
        return self.activations_cache[-1]

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float):
        m = x.shape[0]
        delta = self.activations_cache[-1] - y
        for i in reversed(range(self.num_layers - 1)):
            d_weights = np.dot(self.activations_cache[i].T, delta) / m
            d_biases = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activations[i-1].derivative(self.z_values[i-1])
            self.weights[i] -= learning_rate * d_weights
            self.biases[i] -= learning_rate * d_biases

    def computeLoss(self, y: np.ndarray, output: np.ndarray) -> float:
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), y.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        predictions = np.argmax(self.forward(x), axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)

x_train, y_train, x_test, y_test = loadData()

# Initialize the network with ReLU activations and Softmax output
layer_sizes = [28 * 28, 128, 64, 10]
activations = [ReLu() for _ in range(len(layer_sizes) - 2)] + [Softmax()]

learning_rate = 0.2
epochs = 50

network = NeuralNetwork(layer_sizes, activations)

# Training loop
for epoch in range(epochs):
    output = network.forward(x_train)
    network.backward(x_train, y_train, learning_rate)

    # Print loss and accuracy for each epoch
    train_loss = network.compute_loss(y_train, network.forward(x_train))
    train_accuracy = network.accuracy(x_train, y_train)
    test_accuracy = network.accuracy(x_test, y_test)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")