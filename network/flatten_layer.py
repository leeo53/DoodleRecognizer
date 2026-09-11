
class FlattenLayer:
    def __init__(self):
        self.input_shape = None

    def forward(self, layer_input):
        '''
        flattens the layer input
        :param layer_input:
        :return: layer_input.reshape(layer_input.shape[0], -1)
        '''
        self.input_shape = layer_input.shape
        return layer_input.reshape(layer_input.shape[0], -1)

    def backward(self, grad_output):
        '''
        backward pass
        :param grad_output: gradients of the loss with respect to the output of the layer
        :return: the unflattened gradients of the loss with respect to the output of the layer
        '''
        return grad_output.reshape(self.input_shape)