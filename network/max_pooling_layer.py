import torch

class MaxPoolLayer:
    def __init__(self, kernel_size, stride):
        '''
        :param kernel_size:
        :param stride:
        '''
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_indices = None
        self.layer_input = None

    def forward(self, input):
        '''
        applies max pooling to the input tensor
        :param input:
        :return: pooled tensor
        '''
        self.layer_input = input
        patches = (
            input
            .unfold(2, self.kernel_size, self.stride)
            .unfold(3, self.kernel_size, self.stride)
        )
        patches = patches.contiguous()
        patches_flat = patches.view(
            *patches.shape[:4],
            self.kernel_size * self.kernel_size
        )

        output, self.max_indices = patches_flat.max(dim=-1)
        return output

    def backward(self, grad_output):
        '''
        :param grad_output: the gradients of the loss with respect to the output of the layer
        :return: the gradient of the loss with respect to the input of the layer
        '''
        local_row = self.max_indices // self.kernel_size
        local_col = self.max_indices % self.kernel_size

        B, C, H_out, W_out = grad_output.shape
        W_in = self.layer_input.shape[3]

        h_base = (
                torch.arange(H_out, device=grad_output.device)
                .view(1, 1, H_out, 1)
                * self.stride
        )

        w_base = (
                torch.arange(W_out, device=grad_output.device)
                .view(1, 1, 1, W_out)
                * self.stride
        )

        input_rows = h_base + local_row
        input_cols = w_base + local_col

        flat_indices = input_rows * W_in + input_cols

        grad_input = torch.zeros_like(self.layer_input)

        grad_input.view(B, C, -1).scatter_add_(
            dim=2,
            index=flat_indices.view(B, C, -1),
            src=grad_output.reshape(B, C, -1)
        )

        return grad_input

