import numpy as np
import pickle
from os import listdir
from enum import Enum
from data import img2array
import random


# with np.errstate(all='ignore'):
#     np.float64(1.0) / 0.0

# np.seterr(all='log')

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


# class ActivationFunction:
#     def __init__(self, func, dfunc):
#         self.func = func
#         self.dfunc = dfunc


# sigmoid = ActivationFunction(
#     lambda x: 1 / (1 + np.exp(-x)),
#     lambda y: y * (1 - y)
# )

# tanh = ActivationFunction(
#     lambda x: np.tanh(x),
#     lambda y: 1 - (y * y)
# )

# relu = ActivationFunction(
#     lambda x: np.maximum(0, x),
#     lambda y: 1 if y > 0 else 0
# )

# softmax = ActivationFunction(
#     lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1),
#     lambda y: np.exp(y) / np.sum(np.exp(y), axis=-1)
# )


def _softmax(x):
    return np.exp(x - np.max(x, axis=1)[:, np.newaxis]) / np.sum(
        np.exp(x - np.max(x, axis=1)[:, np.newaxis]), axis=1)[:, np.newaxis]


# softmax = ActivationFunction(
#     lambda x: np.exp(x - np.max(x, axis=1)[:, np.newaxis]) / np.sum(
#         np.exp(x - np.max(x, axis=1)[:, np.newaxis]), axis=1)[:, np.newaxis],
#     lambda y: 0
# )

# np.random.seed(1)


def _randomize(first, second):
    return np.random.rand(first, second)*2-1


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

    def predict(self, input_arr, chance=False, t=False):
        inputs = np.array(input_arr, ndmin=2, dtype="longdouble")

        current = np.dot(inputs, self.weights[0])
        current = np.add(current, self.bias[0], dtype="longdouble")
        current = _sigmoid(current)

        for i in range(1, len(self.weights)):
            current = np.dot(current, self.weights[i])
            current = np.add(current, self.bias[i], dtype="longdouble")
            current = _sigmoid(current)

        current = _softmax(current) if chance else current
        return current

    def train(self, input_arr, target_arr):
        inputs = np.array(input_arr, ndmin=2, dtype="longdouble")
        targets = np.array(target_arr, ndmin=2, dtype="longdouble")
        # inputs = _sigmoid(inputs)
        # targets = _sigmoid(targets)

        outputs = self.predict(input_arr)

        current_error = np.subtract(targets, outputs, dtype="longdouble")

        gradients = _sigmoid(outputs, True)
        gradients = np.multiply(gradients, current_error, dtype="longdouble")
        gradients = np.multiply(
            gradients, self.learning_rate, dtype="longdouble")

        nodes_values = [0 for i in range(len(self.hidden_node))]

        nodes_values[0] = np.dot(inputs, self.weights[0])
        nodes_values[0] = np.add(
            nodes_values[0], self.bias[0], dtype="longdouble")
        nodes_values[0] = _sigmoid(nodes_values[0])

        for i in range(1, len(nodes_values)):
            nodes_values[i] = np.dot(nodes_values[i-1], self.weights[i])
            nodes_values[i] = np.add(
                nodes_values[i], self.bias[i], dtype="longdouble")
            nodes_values[i] = _sigmoid(nodes_values[i])

        current_T = np.transpose(nodes_values[-1])
        weight_deltas = np.dot(current_T, gradients)

        self.weights[-1] = np.add(self.weights[-1],
                                  weight_deltas, dtype="longdouble")
        self.bias[-1] = np.add(self.bias[-1], gradients, dtype="longdouble")

        for i in range(len(nodes_values)-1, -1, -1):

            weights_T = np.transpose(self.weights[i+1])
            current_error = np.dot(current_error, weights_T)

            gradients = _sigmoid(nodes_values[i], True)
            gradients = np.multiply(
                gradients, current_error, dtype="longdouble")
            gradients = np.multiply(
                gradients, self.learning_rate, dtype="longdouble")

            if i == 0:
                current_T = np.transpose(inputs)
                weight_deltas = np.dot(current_T, gradients)

                self.weights[0] = np.add(
                    self.weights[0], weight_deltas, dtype="longdouble")
                self.bias[0] = np.add(
                    self.bias[0], gradients, dtype="longdouble")
            else:
                current_T = np.transpose(nodes_values[i-1])
                weight_deltas = np.dot(current_T, gradients)

                self.weights[i] = np.add(
                    self.weights[i], weight_deltas, dtype="longdouble")
                self.bias[i] = np.add(
                    self.bias[i], gradients, dtype="longdouble")

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


def train_faces(times=1000):
    files = listdir(r'NEW DATA/Faces/Male')
    print("Start Face training")
    for i in range(times):
        rand = random.randint(0, len(files)-1)
        path = r'NEW DATA/Faces/Male/' + files[rand]
        data = img2array(path)
        nn.train(data, [1, 0])
        print("Progress: " + str(i) + "/" + str(times))
    print("Face training ended")


def train_animals(times=1000):
    files = listdir(r'NEW DATA/Faces/Female')
    print('Start Animal training')
    for i in range(times):
        rand = random.randint(0, len(files)-1)
        path = r'NEW DATA/Faces/Female/' + files[rand]
        data = img2array(path)
        nn.train(data, [0, 1])
        print("Progress: " + str(i) + "/" + str(times))
    print("Animal training ended")


def train_both(times=1000):
    faces = listdir(r'NEW DATA/Faces/Male')
    animals = listdir(r'NEW DATA/Faces/Female')
    print('Start training')
    for i in range(times):
        select = random.randint(0, 1)
        if select == 0:
            rand = random.randint(0, len(faces)-1)
            path = r'NEW DATA/Faces/Male/' + faces[rand]
            data = img2array(path)
            nn.train(data, [1, 0])
        else:
            rand = random.randint(0, len(animals)-1)
            path = r'NEW DATA/Faces/Female/' + animals[rand]
            data = img2array(path)
            nn.train(data, [0, 1])
        print('Progress: ' + str(i) + '/' + str(times))


def train_both_max():
    faces = listdir(r'NEW DATA/Faces/Male')
    animals = listdir(r'NEW DATA/Faces/Female')

    face_i = 0
    animal_i = 0

    print('Start training')
    for i in range(len(faces)+len(animals)-2):
        select = random.randint(0, 1)
        if (select == 0 and face_i < len(faces)) or animal_i >= len(animals):
            # rand = random.randint(0, len(faces)-1)
            path = r'NEW DATA/Faces/Male/' + faces[face_i]
            data = img2array(path)
            nn.train(data, [1, 0])
            face_i += 1
        else:
            # rand = random.randint(0, len(animals)-1)
            path = r'NEW DATA/Faces/Female/' + animals[animal_i]
            data = img2array(path)
            nn.train(data, [0, 1])
            animal_i += 1
        print('Progress: ' + str(i) + '/' + str(len(faces)+len(animals)))


w, h = 28, 28

nn = NeuralNetwork(w*h, [100], 2, learning_rate=0.2)

# train_both_max()
train_both(3000)
nn.saveNetwork()

# i = 0
# for w in nn.weights:
#     print(i, w)
#     i += 1


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
