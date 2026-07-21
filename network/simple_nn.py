import numpy as np
from enum import Enum

class Layer_Type(Enum):
    LINEAR = 1
    RELU = 2
    SOFTMAX = 3
    SIGMOID = 4
    TANH = 5


class Layer:
    def __init__(self, input_dim, neurons, layer_type, learning_rate=None):
        self.input_dim = input_dim
        self.neurons = neurons
        self.layer_type = layer_type
        if self.layer_type == Layer_Type.LINEAR:
            self.weights = (
                    np.random.randn(input_dim, neurons)
                    * np.sqrt(2.0 / input_dim)
            )
            self.biases = np.zeros(neurons)
            self.grad_w = None
            self.grad_b = None
        self.lr = learning_rate
        self.layer_input = None
        self.output = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        if self.layer_type == Layer_Type.LINEAR:
            return layer_input.dot(self.weights) + self.biases
        elif self.layer_type == Layer_Type.RELU:
            return np.maximum(0, layer_input)
        elif self.layer_type == Layer_Type.SOFTMAX:
            shifted_input = layer_input - np.max(
                layer_input,
                axis=1,
                keepdims=True
            )
            numerator = np.exp(shifted_input)
            denominator = np.sum(numerator, axis=1, keepdims=True)
            self.output = numerator / denominator
            return self.output
        elif self.layer_type == Layer_Type.SIGMOID:
            self.output = 1 / (1 + np.exp(-layer_input))
            return self.output
        else:
            self.output = np.tanh(layer_input)
            return self.output

    #grad_o is the gradient of the loss with respect to the output of this layer
    #grad_o is y for softmax layers
    def backward(self, grad_o):
        if self.layer_type == Layer_Type.LINEAR:
            grad_w = self.layer_input.T @ grad_o
            grad_b = np.sum(grad_o, axis=0)
            self.grad_w = grad_w
            self.grad_b = grad_b
            grad_l_i = grad_o @ self.weights.T
            self.weights = self.weights - grad_w * self.lr
            self.biases = self.biases - grad_b * self.lr
            return grad_l_i
        elif self.layer_type == Layer_Type.RELU:
            return grad_o * (self.layer_input > 0)
        elif self.layer_type == Layer_Type.SOFTMAX: #combined with cross entropy
            y = grad_o
            return (self.output - y) / y.shape[0]
        elif self.layer_type == Layer_Type.SIGMOID:
            return grad_o * (self.output*(1-self.output))
        else:
            return grad_o * (1 - self.output**2)


            



class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def add_linear_layer(self, output_dim):
        self._add_layer(Layer_Type.LINEAR, output_dim, self.learning_rate)

    def add_ReLU_layer(self):
        self._add_layer(Layer_Type.RELU)

    def add_sigmoid_layer(self):
        self._add_layer(Layer_Type.SIGMOID)

    def add_tanh_layer(self):
        self._add_layer(Layer_Type.TANH)

    def add_softmax_layer(self):
        self._add_layer(Layer_Type.SOFTMAX)

    def forward(self, input):
        z=input
        for i,layer in enumerate(self.layers):
            z= layer.forward(z)
        return z

    def backward(self, output):
        grad_output = output #start with the target labels
        for i in range(len(self.layers) - 1, -1, -1):
            grad_output = self.layers[i].backward(grad_output)

    def fit(self, X, y):
        if len(self.layers) == 0:
            return -1

        softmax_only_last = True

        for i in range(len(self.layers)):
            if (
                    self.layers[i].layer_type == Layer_Type.SOFTMAX
                    and i != len(self.layers) - 1
            ):
                softmax_only_last = False
                break

        if self.layers[-1].layer_type != Layer_Type.SOFTMAX:
            softmax_only_last = False
        elif self.layers[-1].neurons != self.output_size:
            softmax_only_last = False

        if not softmax_only_last:
            return -1

        for epoch in range(self.epochs):
            probs = self.forward(X)
            loss = self.cross_entropy_error(y, probs)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.6f}")

            self.backward(y)

        return 0

    def _add_layer(self, layer_type, output_dim=None, learning_rate=None):
        if output_dim is None:
            if layer_type == Layer_Type.LINEAR:
                self.layers.append(Layer(self.layers[-1].neurons, self.output_size, layer_type, learning_rate))
            else:
                self.layers.append(Layer(self.layers[-1].neurons, self.layers[-1].neurons, layer_type, learning_rate))
        elif len(self.layers) == 0:
            self.layers.append(Layer(input_dim=self.input_size, neurons=output_dim, layer_type=layer_type, learning_rate=learning_rate))
        else:
            self.layers.append(Layer(self.layers[-1].neurons, output_dim, layer_type, learning_rate))

    def cross_entropy_error(self, y, probs):
        probs = np.clip(probs, 1e-15, 1.0)
        return -np.sum(y * np.log(probs))/y.shape[0]

