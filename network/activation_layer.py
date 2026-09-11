
import torch

class ReLULayer:
    def __init__(self):
        self.layer_input = None

    def forward(self, layer_input):
        '''
        forward pass
        :param layer_input:
        :return: torch.clamp(layer_input, min=0)
        '''
        self.layer_input = layer_input
        return torch.clamp(layer_input, min=0)

    def backward(self, grad_o):
        '''
        backward pass
        :param grad_o: gradient of the loss with respect to the output of the layer
        :return: the gradient of the loss with respect to the input of the layer
        '''
        return grad_o * (self.layer_input > 0)

class OutputSoftmaxLayer:
    def __init__(self):
        self.output = None

    def forward(self, layer_input):
        '''
        forward pass
        :param layer_input:
        :return: softmax of layer_input
        '''
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
        '''
        backward pass combined with cross entropy gradient
        :param y: labels for the data
        :return: the gradient of the cross entropy loss with respect to the input to softmax
        '''
        return (self.output - y) / y.shape[0]

class SigmoidLayer:
    def __init__(self):
        self.output = None

    def forward(self, layer_input):
        '''
        forward pass
        :param layer_input:
        :return: sigmoid of layer_input
        '''
        self.output = 1 / (1 + torch.exp(-layer_input))
        return self.output

    def backward(self, grad_o):
        '''
        backward pass
        :param grad_o: gradient of the loss with respect to the output
        :return: gradient of the loss with respect to the input of the layer
        '''
        return grad_o * (self.output * (1 - self.output))

class TanhLayer:
    def __init__(self):
        self.output = None

    def forward(self, layer_input):
        '''
        forward pass
        :param layer_input:
        :return: Tanh of layer_input
        '''
        self.output = torch.tanh(layer_input)
        return self.output

    def backward(self, grad_o):
        '''
        backward pass
        :param grad_o: gradient of the loss with respect to the output
        :return: gradient of the loss with respect to the input of the layer
        '''
        return grad_o * (1 - self.output ** 2)