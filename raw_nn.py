import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

def mse(y, a):
    return np.mean((y - a) ** 2)

def mse_grad(y, a):
    return 2 * (a - y) / np.size(y)

def binary_cross_entropy(y, a):
    return np.mean(-y * np.log(a) - (1 - y) * np.log(1 - a))

def binary_cross_entropy_grad(y, a):
    a += 1e-10
    return ((1 - y) / (1 - a) - y / a) / np.size(y)

def categorical_cross_entropy(y, p):
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return - y * np.log(p) - (1 - y) * np.log(1 - p)

def categorical_cross_entropy_grad(y, p):
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return (y / p) + (1 - y) / (1 - p)

class Relu():
    def forward(self, inp):
        self.input = inp
        self.output = np.maximum(0, inp)
        return self.output

    def backward(self, _, out_grad):
        return np.where(self.input > 0, out_grad, 0)

class Sigmoid():
    def forward(self, inp):
        self.input = inp
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output
    
    def backward(self, _, out_grad):
        return self.output * (1 - self.output) * out_grad

class Softmax():
    def __init__(self, axis):
        self.axis = axis

    def forward(self, inp):
        self.input = inp
        exp_scores = np.exp(inp)
        s = np.sum(exp_scores, axis=self.axis, keepdims=True) + 1e-6
        self.output = exp_scores / s
        return self.output

    def backward(self, _, out_grad):
        n = self.input.shape[0]
        return out_grad * (self.output * (1 - self.output)) / n

class LeakyRelu():
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, inp):
        self.input = inp
        self.output = np.maximum(self.alpha * inp, inp)
        return self.output

    def backward(self, _, out_grad):
        return np.where(self.input > 0, out_grad, self.alpha * out_grad)

class Tanh():
    def forward(self, inp):
        self.input = inp
        self.output = np.tanh(inp)
        return self.output

    def backward(self, _, out_grad):
        return out_grad * (1 - self.output ** 2)

class Transpose():
    def __init__(self, axis):
        self.axis = axis

    def forward(self, inp):
        self.input = inp
        self.output = np.transpose(inp, self.axis)
        return self.output

    def backward(self, _, out_grad):
        return np.transpose(out_grad, self.axis)

class Dense():
    def __init__(self, input_size, output_size):
        limit = 1 / np.sqrt(input_size)
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.bias = np.random.randn(output_size, 1)

    def forward(self, inp):
        self.input = inp
        bias = np.repeat(self.bias, self.input.shape[1], axis=1) if len(self.input.shape) > 1 else self.bias
        self.output = np.dot(self.weights, self.input) + bias
        return self.output

    def backward(self, learning_rate, out_grad):
        grad_i = np.dot(self.weights.T, out_grad)
        grad_w = np.dot(out_grad, self.input.T) / self.input.shape[0]
        grad_b = np.sum(out_grad, axis=1, keepdims=True) / self.input.shape[0]
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b
        return grad_i

def predict(x, layers):
    a = x
    for layer in layers:
        a = layer.forward(a)
    return a

def train_xor():
    layers = [
        Dense(4, 3),
        LeakyRelu(0.3),
        Dense(3, 4),
        Softmax(axis=1),
    ]
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    y_ = np.reshape(np.eye(2)[y], (4, -1))
    for i in range(10000):
        a = predict(x, layers)
        preds = np.argmax(a, axis=1)
        # accuracy = np.mean(preds == np.reshape(y, -1))
        grad = binary_cross_entropy_grad(y_, a)

        for layer in reversed(layers):
            grad = layer.backward(0.5, grad)
        if i % 1000 == 0:
            print(a)

def train_add():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    y_ = np.reshape(np.eye(2)[y], (4, -1))
    layers = [
        Dense(2, 3),
        Tanh(),
        Dense(3, 2),
        Softmax(axis=0),
    ]
    for i in range(10000):
        a = predict(x.T, layers)
        preds = np.argmax(a, axis=1)
        accuracy = np.mean(preds == np.reshape(y, -1))
        grad = binary_cross_entropy_grad(y_, a.T).T
        for layer in reversed(layers):
            grad = layer.backward(0.5, grad)
        if i % 1000 == 0:
            print(a)

train_xor()
train_add()
