import numpy as np
from typing import List

import utils.base as base

import utils.data_loaders as dl
import utils.activation_functions as af
import utils.loss_functions as lf
import utils.weight_initializers as wi
import utils.bias_initializers as bi
import utils.batch_splitter as bs


# Simple one-layer neural network
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_function: base.ActivationFunction, loss_function: base.LossFunction, weight_initializer: base.WeightInitializer, bias_initializer: base.BiasInitializer):
        # Initialize weights and biases
        self.W1 = weight_initializer.initialize(input_size, hidden_size)
        self.b1 = bias_initializer.initialize(hidden_size)
        self.W2 = weight_initializer.initialize(hidden_size, output_size)
        self.b2 = bias_initializer.initialize(output_size)

        self.activation_function = activation_function
        self.loss_function = loss_function

        self.softmax = af.Softmax()

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
    data_loader = dl.MNIST()
    x_train, y_train, x_test, y_test = data_loader.load()

    # Define hyperparameters
    learning_rate = 0.002
    num_epochs = 100

    batch_splitter = bs.Stochastic()

    nn = NeuralNetwork(input_size=784, hidden_size=10, output_size=10, activation_function=af.ELU(), loss_function=lf.MSE(), weight_initializer=wi.Xavier(), bias_initializer=bi.Zero())

    # Train the network with mini-batch gradient descent
    for epoch in range(1, num_epochs + 1):
        mini_batches = batch_splitter.split(x_train, y_train)

        for x_mini_batch, y_mini_batch in mini_batches:
            output = nn.forward(x_mini_batch)
            nn.backward(x_mini_batch, y_mini_batch, output, learning_rate)

        print(f'Epoch {epoch}')
    
    # Predict and evaluate on the test set
    predictions = np.argmax(nn.forward(x_test), axis=1)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    print(f'Test accuracy: {accuracy * 100:.2f}%')