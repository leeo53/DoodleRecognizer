
from network.convolution_layer import ConvLayer
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_padding_one_in_channel():
    layer = ConvLayer(1,3,3,2,1,0)
    image = torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    ], dtype=torch.float32, device=device)
    correct_result = torch.tensor([
        [
            [0,0,0,0,0],
            [0,1,2,3,0],
            [0,4,5,6,0],
            [0,7,8,9,0],
            [0,0,0,0,0]

        ]
    ], dtype=torch.float32, device=device)
    correct_result = correct_result.unsqueeze(0)
    images = image.unsqueeze(0)
    padded_image = layer.apply_padding(images)
    assert torch.equal(padded_image, correct_result)

def test_padding_two_in_channels():
    layer = ConvLayer(2,3,3,2,1,0)
    images = torch.tensor([
        [  # Image 1
            [  # Channel 0
                [1, 2],
                [3, 4]
            ],
            [  # Channel 1
                [10, 20],
                [30, 40]
            ]
        ]
    ], dtype=torch.float32, device=device)
    correct_result = torch.tensor([
        [  # Image 1
            [  # Channel 0
                [0,0,0,0],
                [0,1,2,0],
                [0,3,4,0],
                [0,0,0,0]
            ],
            [  # Channel 1
                [0, 0, 0, 0],
                [0, 10, 20, 0],
                [0, 30, 40, 0],
                [0, 0, 0, 0]
            ]
        ]
    ], dtype=torch.float32, device=device)
    padded_images = layer.apply_padding(images)
    assert torch.equal(padded_images, correct_result)

def test_padding_zero():
    layer = ConvLayer(2,3,3,2,0,0)
    images = torch.tensor([
        [  # Image 1
            [  # Channel 0
                [1, 2],
                [3, 4]
            ],
            [  # Channel 1
                [10, 20],
                [30, 40]
            ]
        ]
    ], dtype=torch.float32, device=device)

    padded_images = layer.apply_padding(images)
    assert torch.equal(padded_images, images)

def test_kernel_initializer():
    layer = ConvLayer(2,3,3,2,1,0)
    kernels, biases = layer.kernels, layer.biases
    assert kernels.size() == (3,2,3,3) and biases.size() == (3,)
    assert kernels.dtype == torch.float32
    assert biases.dtype == torch.float32

    assert torch.all(biases == 0)

    assert kernels.device.type == device.type
    assert biases.device.type == device.type

def test_forward_basic():
    layer = ConvLayer(1,1,3,1,1, learning_rate=0)
    image = torch.tensor([
        [
            [1, 2, 3,4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ], dtype=torch.float32, device=device)
    images = image.unsqueeze(0)
    layer.kernels = torch.tensor([
        [
            [1,0,-1],
            [1,0,-1],
            [1,0,-1]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    layer.biases = torch.tensor([2.0],dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.tensor([
        [
            [-6,-2,-2,12],
            [-16,-4,-4,23],
            [-28,-4,-4,35],
            [-22,-2,-2,28]
        ]
    ],dtype=torch.float32, device=device).unsqueeze(0)
    assert torch.equal(result_maps, correct_maps)

def test_forward_no_padding():
    layer = ConvLayer(1, 1, 3, 1, 0, learning_rate=0)
    image = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ], dtype=torch.float32, device=device)
    images = image.unsqueeze(0)
    layer.kernels = torch.tensor([
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    layer.biases = torch.tensor([2.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.tensor([
        [
            [-4, -4],
            [-4, -4]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    assert torch.equal(result_maps, correct_maps)

def test_forward_2_stride():
    layer = ConvLayer(1,1,3,2,1, learning_rate=0)
    image = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]
    ], dtype=torch.float32, device=device)
    images = image.unsqueeze(0)
    layer.kernels = torch.tensor([
        [
            [1, 0, -1],
            [1, -1, 0],
            [1, 0, -1]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    layer.biases = torch.tensor([2.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.tensor([
        [
            [-5, -1],
            [-27, -3]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    assert torch.equal(result_maps, correct_maps)

def test_forward_multiple_in_channels():
    layer = ConvLayer(3,1,3,1,1, learning_rate=0)
    image = torch.tensor([
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [15, 8, 1, 5],
            [14, 9, 2, 6],
            [13, 10, 3, 7],
            [12, 11, 4, 8]
        ],
        [
            [10,20,50,60],
            [30,40,70,80],
            [90,100,130,140],
            [110,120,150,160]
        ]
    ], dtype=torch.float32, device=device)
    images = image.unsqueeze(0)
    layer.kernels = torch.tensor([
        [
            [1, 0, -1],
            [1, -1, 0],
            [1, 0, -1]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    layer.biases = torch.tensor([2.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.conv2d(images,layer.kernels,layer.biases,1,1)

    assert torch.equal(result_maps, correct_maps)

def test_forward_multiple_out_channels():
    torch.manual_seed(42)
    layer = ConvLayer(1, 2, 3, 1, 1, learning_rate=0)
    images = torch.randn(1,1,4,4, device=device)
    layer.kernels = torch.randn(2,1,3,3, device=device)
    layer.biases = torch.tensor([2.0,1.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.conv2d(images, layer.kernels, layer.biases, 1, 1)

    torch.testing.assert_close(result_maps, correct_maps)

def test_forward_multiple_in_and_out_channels():
    torch.manual_seed(42)
    layer = ConvLayer(3, 2, 3, 1, 1, learning_rate=0)
    images = torch.randn(1, 3, 4, 4, device=device)
    layer.kernels = torch.randn(2, 3, 3, 3, device=device)
    layer.biases = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.conv2d(images, layer.kernels, layer.biases, 1, 1)

    torch.testing.assert_close(result_maps, correct_maps)

def test_forward_multiple_images():
    torch.manual_seed(42)
    layer = ConvLayer(3, 2, 3, 1, 1, learning_rate=0)
    images = torch.randn(5, 3, 4, 4, device=device)
    layer.kernels = torch.randn(2, 3, 3, 3, device=device)
    layer.biases = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device)
    result_maps = layer.forward(images)
    correct_maps = torch.conv2d(images, layer.kernels, layer.biases, 1, 1)

    torch.testing.assert_close(result_maps, correct_maps)

def test_forward_output_shape():
    torch.manual_seed(42)
    for in_channels in [1,3]:
        for out_channels in [1,2,3]:
            for kernel_size in [2,3]:
                for stride in [1,2]:
                    for padding in [0,1]:
                        for batch_size in [1,2]:
                            layer = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, learning_rate=0)
                            images = torch.randn(batch_size, in_channels, 4, 4, device=device)
                            layer.kernels = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
                            layer.biases = torch.randn(out_channels, device=device)
                            result_maps = layer.forward(images)
                            H_out = (images.shape[-2] + (2*padding) - kernel_size)// stride + 1
                            W_out = (images.shape[-1] + (2*padding) - kernel_size)// stride + 1
                            assert result_maps.shape == (batch_size,out_channels,H_out, W_out)

def test_bias_broadcasting():
    torch.manual_seed(42)
    layer = ConvLayer(3, 3, 3, 1, 1, learning_rate=0)
    images = torch.randn(1,3,4,4, device=device)
    layer.kernels = torch.randn(3,3,3,3, device=device)
    biases = torch.randn(3,device=device)
    layer.biases = biases
    result_maps_with_b = layer.forward(images)
    layer.biases = torch.zeros(3,device=device, dtype=torch.float32)
    result_maps_without = layer.forward(images)
    torch.testing.assert_close(result_maps_with_b, result_maps_without + biases[None, :, None, None])

def test_backward_basic():
    _check_backward(3, 3, 3, 1, 1, 1, 4,4)

def test_backward_stride_greater_than_1():
    _check_backward(3, 3, 3, 2, 1, 1, 4,4)

def test_backward_one_input_channel():
    _check_backward(1, 1, 3, 1, 1, 1, 4,4)

def test_backward_padding_0():
    _check_backward(3, 2, 2, 1, 0, 1, 4,4)

def test_backward_more_images():
    _check_backward(3, 1, 2, 1, 1, 3, 6,6)

def test_backward_non_square_images():
    _check_backward(3, 1, 2, 1, 1, 3, 4,6)

def test_parameter_update():
    torch.manual_seed(42)
    lr = .05
    layer = ConvLayer(3, 3, 3, 2, 1, learning_rate=lr)
    images = torch.randn(1, 3, 4, 4, device=device)
    kernels = torch.randn(3, 3, 3, 3, device=device)
    old_kernels = kernels.clone()
    layer.kernels = kernels
    biases = torch.randn(3, device=device)
    old_biases = biases.clone()
    layer.biases = biases
    result_maps = layer.forward(images)
    output_grad = torch.randn_like(result_maps)
    layer.backward(output_grad)
    result_kernel_grad = layer.grad_kernels
    result_bias_grad = layer.grad_biases
    torch.testing.assert_close(old_kernels - (lr * result_kernel_grad) , layer.kernels)
    torch.testing.assert_close(old_biases - (lr * result_bias_grad) , layer.biases)

def test_backward_output_shape():
    torch.manual_seed(42)
    layer = ConvLayer(3, 3, 3, 1, 1, learning_rate=0)
    images = torch.randn(1, 3, 4, 4, device=device)
    layer.kernels = torch.randn(3,3,3,3, device=device)
    layer.biases = torch.randn(3,device=device)
    result_maps = layer.forward(images)
    output_grad = torch.randn_like(result_maps)
    result_images_grad = layer.backward(output_grad)
    assert result_images_grad.shape == images.shape

def _check_backward(in_channels, out_channels, kernel_size, stride, padding, batch_size, height, width):
    torch.manual_seed(42)
    layer = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, learning_rate=0)

    images = torch.randn(batch_size, in_channels, height, width, device=device)
    kernels = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device)
    biases = torch.randn(out_channels, device=device)

    layer.kernels = kernels
    layer.biases = biases

    result_maps = layer.forward(images)
    output_grad = torch.randn_like(result_maps)
    result_images_grad = layer.backward(output_grad)

    pt_images = images.clone().requires_grad_(True)
    pt_kernels = kernels.clone().requires_grad_(True)
    pt_biases = biases.clone().requires_grad_(True)

    output = F.conv2d(
        pt_images,
        pt_kernels,
        pt_biases,
        stride,
        padding
    )
    output.backward(output_grad)

    torch.testing.assert_close(
        result_images_grad,
        pt_images.grad
    )

    torch.testing.assert_close(
        layer.grad_biases,
        pt_biases.grad
    )

    torch.testing.assert_close(
        layer.grad_kernels,
        pt_kernels.grad
    )



