import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearLayer:
    def __init__(self, input_dim, neurons, learning_rate):
        self.input_dim = input_dim
        self.neurons = neurons
        self.weights = (
                    torch.randn(input_dim, neurons, device=device)
                    * (2.0 / input_dim) ** 0.5
            )
        self.biases = torch.zeros(neurons, device=device)
        self.grad_w = None
        self.grad_b = None
        self.lr = learning_rate
        self.layer_input = None

    def forward(self, layer_input):
        self.layer_input = layer_input
        return layer_input @ self.weights + self.biases

    def backward(self, grad_o):
        grad_w = self.layer_input.T @ grad_o
        grad_b = torch.sum(grad_o, dim=0)
        self.grad_w = grad_w
        self.grad_b = grad_b
        grad_l_i = grad_o @ self.weights.T
        self.weights -= self.lr * grad_w
        self.biases -= self.lr * grad_b
        return grad_l_i