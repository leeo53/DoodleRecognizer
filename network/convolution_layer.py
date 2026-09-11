from math import sqrt

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLayer:
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding, learning_rate):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.kernels, self.biases = self.initialize_kernels()
        self.lr = learning_rate
        self.padded_input_images = None
        self.feature_maps = None
        self.grad_kernels = None
        self.grad_biases = None
        self.patches = None

    def initialize_kernels(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        std = sqrt(2 / fan_in)
        kernels=torch.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            device=device,
            dtype=torch.float32) * std
        biases = torch.zeros(self.out_channels, device=device, dtype=torch.float32)
        return kernels,biases

    def forward(self, images):
        padded_images = self.apply_padding(images)
        self.padded_input_images = padded_images
        self.patches = F.unfold(
            self.padded_input_images,
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        flat_kernels = self.kernels.reshape(self.out_channels, -1)
        result = flat_kernels @ self.patches
        output_height = (
                                (self.padded_input_images.shape[2] - self.kernel_size)
                                // self.stride
                        ) + 1

        output_width = (
                               (self.padded_input_images.shape[3] - self.kernel_size)
                               // self.stride
                       ) + 1
        result = result.reshape(
            self.padded_input_images.shape[0],
            self.out_channels,
            output_height,
            output_width,
        )
        self.feature_maps = result + self.biases[None,:,None,None]
        return self.feature_maps

    def apply_padding(self, images):
        if self.padding != 0:
            padded = torch.zeros(
                images.shape[0],
                self.in_channels,
                images.shape[2]+(self.padding*2),
                images.shape[3]+(self.padding*2),
                device=device,
                dtype = torch.float32)
            padded[:,:,self.padding:-self.padding,self.padding:-self.padding] = images
            return padded
        else:
            return images

    def backward(self,gradients):
        gradients_wrt_biases = gradients.sum(dim=(0,2,3))

        flat_gradients = gradients.reshape(
            gradients.shape[0],
            self.out_channels,
            -1
        )

        flat_gradient_wrt_kernels = (flat_gradients @ self.patches.transpose(1,2)).sum(dim=0)

        gradient_wrt_kernels = flat_gradient_wrt_kernels.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        s = self.stride
        k = self.kernel_size
        flat_kernels = self.kernels.reshape(self.out_channels, -1)
        contributions = flat_kernels.T @ flat_gradients

        gradients_wrt_padded_input = F.fold(
            contributions,
            (self.padded_input_images.shape[2],self.padded_input_images.shape[3]),
            kernel_size = k,
            stride = s
        )
        self.grad_kernels = gradient_wrt_kernels
        self.grad_biases = gradients_wrt_biases

        self.kernels = self.kernels - gradient_wrt_kernels * self.lr
        self.biases = self.biases - gradients_wrt_biases * self.lr

        if self.padding > 0:
            gradients_wrt_input = gradients_wrt_padded_input[
                :, :,
                self.padding:-self.padding,
                self.padding:-self.padding
            ]
        else:
            gradients_wrt_input = gradients_wrt_padded_input

        return gradients_wrt_input