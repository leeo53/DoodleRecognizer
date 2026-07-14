import numpy as np
from enum import Enum

class Layer_Type(Enum):
    LINEAR = 1
    RELU = 2
    SOFTMAX = 3
    SIGMOID = 4
    TANH = 5


class Layer:
    def __init__(self, input_dim, neurons, layer_type):
        self.input_dim = input_dim
        self.neurons = neurons
        self.layer_type = layer_type
        self.weights = np.random.randn(input_dim, neurons)
        self.biases = np.ones(neurons)
        self.layer_input = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        if self.layer_type == Layer_Type.LINEAR:
            return layer_input.dot(self.weights)
        elif self.layer_type == Layer_Type.RELU:
            return np.maximum(0, layer_input)
        elif self.layer_type == Layer_Type.SOFTMAX:
            return np.exp(layer_input) / np.sum(np.exp(layer_input))
        elif self.layer_type == Layer_Type.SIGMOID:
            return 1 / (1 + np.exp(-layer_input))
        else:
            return np.tanh(layer_input)

    #grad_o is the gradient of the loss with respect to the output of this layer
    def backward(self, grad_o):
        if self.layer_type == Layer_Type.LINEAR:
            grad_w = self.layer_input.T @ grad_o
            grad_b = np.sum(grad_o, axis=0)
            grad_l_i = grad_o @ self.weights.T
            self.weights = self.weights + grad_w * self.lr
            self.biases = self.biases + grad_b * self.lr
            return grad_l_i
        if self.layer_type == Layer_Type.RELU:
            



class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def add_linear_layer(self, output_dim=None):
        self._add_layer(Layer_Type.LINEAR, output_dim)

    def add_ReLU_layer(self, output_dim=None):
        self._add_layer(Layer_Type.RELU, output_dim)

    def add_sigmoid_layer(self, output_dim=None):
        self._add_layer(Layer_Type.SIGMOID, output_dim)

    def add_tanh_layer(self, output_dim=None):
        self._add_layer(Layer_Type.TANH, output_dim)

    def forward(self, input):
        z=input
        for i,layer in enumerate(self.layers):
            z= layer.forward(z)
        return z

    def fit(self, X, y, lr=0.01, epochs=100):
        for i in range(epochs):
            probs = self.forward(X)

    def _add_layer(self, layer_type, output_dim=None):
        if output_dim is None:
            self.layers.append(Layer(self.layers[len(self.layers) - 1].neurons, self.output_size, layer_type))
        elif len(self.layers) == 0:
            self.layers.append(Layer(input_dim=self.input_size, neurons=output_dim, layer_type=layer_type))
        else:
            self.layers.append(Layer(self.layers[len(self.layers) - 1].neurons, output_dim, layer_type))

    def cross_entropy_error(self, y, probs):
        return -np.sum(y * np.log(probs))/y.shape[0]

