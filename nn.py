import numpy as np
from typing import List

from utils.activation_functions import ActivationFunction, ReLu, Softmax
from utils.loss_functions import LossFunction, MeanSquaredError

from utils.load_data import loadMNIST

# Simple one-layer neural network
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_function: ActivationFunction, loss_function: LossFunction):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.activation_function = activation_function
        self.loss_function = loss_function

        self.softmax = Softmax()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Forward pass
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation_function.activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.softmax.activate(self.z2)

    def backward(self, x: np.ndarray, y: np.ndarray, output: np.ndarray, learning_rate: float):
        # Backpropagation
        m = y.shape[0]
        dz2 = self.loss_function.gradient(y, output)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.W2.T) * self.activation_function.derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = loadMNIST()

    learning_rate = 0.05
    num_epochs = 1000

    nn = NeuralNetwork(input_size=784, hidden_size=20, output_size=10, activation_function=ReLu(), loss_function=MeanSquaredError())

    # Train the network
    for epoch in range(1, num_epochs + 1):
            output = nn.forward(x_train)
            nn.backward(x_train, y_train, output, learning_rate)
            print(f'Epoch {epoch}')
    
    # Predict and evaluate on the test set
    predictions = np.argmax(nn.forward(x_test), axis=1)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    print(f'Test accuracy: {accuracy * 100:.2f}%')