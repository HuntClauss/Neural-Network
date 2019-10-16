import numpy as np
import pickle
from enum import Enum


def _sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1/(1 + np.exp(-x))


def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1)


def _randomize(first, second):
    return np.random.rand(first, second)*2-1


# np.random.seed(2)


class NeuralNetwork:

    def __init__(self, a, b, c):
        self.input_node = a
        self.hidden_node = b
        self.output_node = c

        self.layer = 2 + len(self.hidden_node)
        self.nodes = [0 for i in range(self.layer)]

        self.nodes[0] = a
        self.nodes[self.layer-1] = c
        for i in range(1, self.layer-1):
            self.nodes[i] = self.hidden_node[i-1]

        self.learning_rate = 0.2

        self.weights = [_randomize(self.nodes[i], self.nodes[i + 1])
                        for i in range(self.layer - 1)]
        self.bias = [_randomize(1, self.nodes[i + 1])
                     for i in range(self.layer - 1)]

    def predict(self, input_arr):
        inputs = np.array(input_arr, ndmin=2)

        current = np.dot(inputs, self.weights[0])
        current = np.add(current, self.bias[0])
        current = _sigmoid(current)

        for i in range(1, len(self.weights)):
            current = np.dot(current, self.weights[i])
            current = np.add(current, self.bias[i])
            current = _sigmoid(current)

        return current

    def train(self, input_arr, target_arr):
        inputs = np.array(input_arr, ndmin=2)
        targets = np.array(target_arr, ndmin=2)

        outputs = self.predict(input_arr)

        current_error = np.subtract(targets, outputs)

        gradients = _sigmoid(outputs, True)
        gradients = np.multiply(gradients, current_error)
        gradients = np.multiply(gradients, self.learning_rate)

        nodes_values = [0 for i in range(len(self.hidden_node))]

        nodes_values[0] = np.dot(inputs, self.weights[0])
        nodes_values[0] = np.add(nodes_values[0], self.bias[0])
        nodes_values[0] = _sigmoid(nodes_values[0])

        for i in range(1, len(nodes_values)):
            nodes_values[i] = np.dot(nodes_values[i-1], self.weights[i])
            nodes_values[i] = np.add(nodes_values[i], self.bias[i])
            nodes_values[i] = _sigmoid(nodes_values[i])

        current_T = np.transpose(nodes_values[len(nodes_values)-1])
        weight_deltas = np.dot(current_T, gradients)

        self.weights[len(
            self.weights)-1] = np.add(self.weights[len(self.weights)-1], weight_deltas)
        self.bias[len(self.bias) -
                  1] = np.add(self.bias[len(self.bias)-1], gradients)

        for i in range(len(nodes_values)-1, -1, -1):

            weights_T = np.transpose(self.weights[i+1])
            current_error = np.dot(current_error, weights_T)

            gradients = _sigmoid(nodes_values[i], True)
            gradients = np.multiply(gradients, current_error)
            gradients = np.multiply(gradients, self.learning_rate)

            if i == 0:
                current_T = np.transpose(inputs)
                weight_deltas = np.dot(current_T, gradients)

                self.weights[0] = np.add(self.weights[0], weight_deltas)
                self.bias[0] = np.add(self.bias[0], gradients)
            else:
                current_T = np.transpose(nodes_values[i-1])
                weight_deltas = np.dot(current_T, gradients)

                self.weights[i] = np.add(self.weights[i], weight_deltas)
                self.bias[i] = np.add(self.bias[i], gradients)

    def saveNetwork(self):
        with open("brain.file", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadNetwork():
        with open("brain.file", "rb") as f:
            return pickle.load(f)


nn = NeuralNetwork(2, [50], 1)

print(nn.predict([1, 1]))
print(nn.predict([0, 0]))
print(nn.predict([1, 0]))
print(nn.predict([0, 1]))

for _ in range(100000):
    nn.train([1, 1], [1])
    nn.train([0, 0], [1])
    nn.train([1, 0], [0])
    nn.train([0, 1], [0])

print("---------------------------")
print(nn.predict([1, 1]))
print(nn.predict([0, 0]))
print(nn.predict([1, 0]))
print(nn.predict([0, 1]))
