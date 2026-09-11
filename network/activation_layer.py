
import torch

class ReLULayer:
    def __init__(self):
        self.layer_input = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        return torch.clamp(layer_input, min=0)

    def backward(self, grad_o):
        return grad_o * (self.layer_input > 0)

class OutputSoftmaxLayer:
    def __init__(self):
        self.layer_input = None
        self.output = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        shifted_input = layer_input - torch.max(
            layer_input,
            dim=1,
            keepdim=True
        ).values
        numerator = torch.exp(shifted_input)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        self.output = numerator / denominator
        return self.output

    def backward(self, y):
        #combined with cross entropy
        return (self.output - y) / y.shape[0]

class SigmoidLayer:
    def __init__(self):
        self.layer_input = None
        self.output = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        self.output = 1 / (1 + torch.exp(-layer_input))
        return self.output

    def backward(self, grad_o):
        return grad_o * (self.output * (1 - self.output))

class TanhLayer:
    def __init__(self):
        self.layer_input = None
        self.output = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        self.output = torch.tanh(layer_input)
        return self.output

    def backward(self, grad_o):
        return grad_o * (1 - self.output ** 2)