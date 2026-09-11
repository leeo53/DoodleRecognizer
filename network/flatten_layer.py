
class FlattenLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, layer_input):
        self.input_shape = layer_input.shape
        return layer_input.reshape(layer_input.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)