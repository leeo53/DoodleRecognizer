
import torch

from network.activation_layer import ReLULayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_relu_sets_negative_values_to_zero():
    relu = ReLULayer()

    X = torch.tensor([[1,2,-3],
                  [4,-5,-6],
                  [-7,8,-9]],device=device, dtype=torch.float32)
    result = relu.forward(X)
    expected = torch.tensor([[1,2,0],
                         [4,0,0],
                         [0,8,0]], device=device, dtype=torch.float32)

    assert torch.equal(result, expected)

def test_relu_preserves_input_shape():
    relu = ReLULayer()
    X = torch.tensor([[1,2,-3]],device=device, dtype=torch.float32)

    result = relu.forward(X)

    assert result.shape == X.shape

def test_relu_backward_blocks_gradient_for_negative_inputs():
    relu = ReLULayer()

    X = torch.tensor([[1,2,-3]], device=device, dtype=torch.float32)

    relu.forward(X)

    upstream_gradient = torch.tensor([[10,20,30]], device=device, dtype=torch.float32)

    result = relu.backward(upstream_gradient)
    expected = torch.tensor([[10,20,0]], device=device, dtype=torch.float32)

    assert torch.equal(result, expected)

def test_relu_backward_preserves_gradient_for_positive_inputs():
    relu = ReLULayer()

    X = torch.tensor(
        [[1.0, 2.0, -3.0]],
        device=device
    )

    relu.forward(X)

    upstream_gradient = torch.tensor(
        [[-10.0, 20.0, -30.0]],
        device=device
    )

    result = relu.backward(upstream_gradient)

    expected = torch.tensor(
        [[-10.0, 20.0, 0.0]],
        device=device
    )

    assert torch.equal(result, expected)

def test_relu_backward_blocks_gradient_at_zero():
    relu = ReLULayer()

    X = torch.tensor([[0.0]], device=device)
    relu.forward(X)

    result = relu.backward(
        torch.tensor([[10.0]], device=device)
    )

    expected = torch.tensor([[0.0]], device=device)

    assert torch.equal(result, expected)