import numpy as np
import pickle
from os import listdir
from enum import Enum
from data import img2array
import random

# def normalize(v):
#     norm = np.linalg.norm(v, ord=1)
#     if norm == 0:
#         norm = np.finfo(v.dtype).eps
#     return v/norm


def _sigmoid(x, deriv=False):
    sigm = 1. / (1. + np.exp(-x))
    if deriv:
        return sigm * (1. - sigm)
    return sigm


def _softmax(x):
    return np.exp(x - np.max(x, axis=1)[:, np.newaxis]) / np.sum(
        np.exp(x - np.max(x, axis=1)[:, np.newaxis]), axis=1)[:, np.newaxis]


def _relu(x, deriv=False):
    # RELU function
    result = x
    result[result <= 0] = 0
    if deriv:
        result[result > 0] = 1
    return result


def _leaky_relu(x, deriv=False):
    result = np.where(x > 0, x, x * 0.02)
    if deriv:
        result[result <= 0] = 0
        result[result > 0] = 1
    return result


def _activation_func(x, deriv=False):
    return _leaky_relu(x, deriv)


def _randomize(first, second):
    return np.random.rand(first, second)


class NeuralNetwork:

    def __init__(self, a, b, c, learning_rate=0.1):
        self.input_node = a
        self.hidden_node = b
        self.output_node = c

        self.layer = 2 + len(self.hidden_node)
        self.nodes = [0 for i in range(self.layer)]

        self.nodes[0] = a
        self.nodes[-1] = c
        for i in range(1, self.layer-1):
            self.nodes[i] = self.hidden_node[i-1]

        self.learning_rate = learning_rate

        self.weights = [_randomize(self.nodes[i], self.nodes[i + 1])
                        for i in range(self.layer - 1)]
        self.bias = [_randomize(1, self.nodes[i + 1])
                     for i in range(self.layer - 1)]

    def predict(self, input_arr, chance=False):
        inputs = np.array(input_arr, ndmin=2, dtype="longdouble")

        current = np.dot(inputs, self.weights[0])
        current = np.add(current, self.bias[0])

        for i in range(1, len(self.weights)):
            current = np.dot(current, self.weights[i])
            current = np.add(current, self.bias[i])

        current = _softmax(current) if chance else _sigmoid(current)
        return current

    def train(self, input_arr, target_arr):
        inputs = np.array(input_arr, ndmin=2, dtype="longdouble")
        targets = np.array(target_arr, ndmin=2, dtype="longdouble")

        outputs = self.predict(input_arr)

        current_error = np.subtract(targets, outputs)

        gradients = _activation_func(outputs, True)
        gradients = np.multiply(gradients, current_error)
        gradients = np.multiply(
            gradients, self.learning_rate)

        nodes_values = [0 for i in range(len(self.hidden_node))]

        nodes_values[0] = np.dot(inputs, self.weights[0])
        nodes_values[0] = np.add(
            nodes_values[0], self.bias[0])
        nodes_values[0] = _activation_func(nodes_values[0])

        for i in range(1, len(nodes_values)):
            nodes_values[i] = np.dot(nodes_values[i-1], self.weights[i])
            nodes_values[i] = np.add(
                nodes_values[i], self.bias[i])
            nodes_values[i] = _activation_func(nodes_values[i])

        current_T = np.transpose(nodes_values[-1])
        weight_deltas = np.dot(current_T, gradients)

        self.weights[-1] = np.add(self.weights[-1],
                                  weight_deltas)
        self.bias[-1] = np.add(self.bias[-1], gradients)

        for i in range(len(nodes_values)-1, -1, -1):

            weights_T = np.transpose(self.weights[i+1])
            current_error = np.dot(current_error, weights_T)

            gradients = _activation_func(nodes_values[i], True)
            gradients = np.multiply(
                gradients, current_error)
            gradients = np.multiply(
                gradients, self.learning_rate)

            if i == 0:
                current_T = np.transpose(inputs)
                weight_deltas = np.dot(current_T, gradients)

                self.weights[0] = np.add(
                    self.weights[0], weight_deltas)
                self.bias[0] = np.add(
                    self.bias[0], gradients)
            else:
                current_T = np.transpose(nodes_values[i-1])
                weight_deltas = np.dot(current_T, gradients)

                self.weights[i] = np.add(
                    self.weights[i], weight_deltas)
                self.bias[i] = np.add(
                    self.bias[i], gradients)

        for i in range(len(self.weights)-1):
            self.weights[i] = _sigmoid(self.weights[i])
            self.bias[i] = _sigmoid(self.bias[i])

    def saveNetwork(self):
        with open("brain.txt", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadNetwork():
        with open("brain.txt", "rb") as f:
            return pickle.load(f)


def train_both(times=1000):
    male = listdir(r'NEW DATA/Faces/Male')
    female = listdir(r'NEW DATA/Faces/Female')
    print('Start training')
    for i in range(times):
        select = random.randint(0, 1)
        if select == 0:
            rand = random.randint(0, len(male)-1)
            path = r'NEW DATA/Faces/Male/' + male[rand]
            data = img2array(path)
            nn.train(data, [1, 0])
        else:
            rand = random.randint(0, len(female)-1)
            path = r'NEW DATA/Faces/Female/' + female[rand]
            data = img2array(path)
            nn.train(data, [0, 1])
        print('Progress: ' + str(i) + '/' + str(times))


nn = NeuralNetwork(28*28, [10000], 1, learning_rate=0.2)

train_both(2)
nn.saveNetwork()

# nn = NeuralNetwork.loadNetwork()

print()
print()
print()

path = r'male.jpg'
data = img2array(path)
result = nn.predict(data, chance=True)
print('Male Test  ', result)

path = r'Female.jpg'
data = img2array(path)
result = nn.predict(data, chance=True)
print('Female Test', result)
