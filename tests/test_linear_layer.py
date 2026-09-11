import torch

from network.linear_layer import LinearLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_linear_layer_forward_calculates_expected():
    layer = LinearLayer(2,2,.05)

    layer.weights = torch.tensor([
        [3.0,4.0],
        [5.0,6.0],
    ], device = device)

    layer.biases = torch.tensor([1.0,2.0], device = device)

    X = torch.tensor([[1.0,2.0]],device = device)

    result = layer.forward(X)

    expected = torch.tensor([[14.0,18.0]],device = device)

    torch.testing.assert_close(result, expected)

def test_linear_layer_returns_correct_shape():
    layer = LinearLayer(3,4,.05)
    X = torch.ones((5,3), device = device)
    result = layer.forward(X)
    assert result.shape == (5,4)

def test_linear_layer_weight_gradient_matches_numerical_gradient():
    layer = LinearLayer(2,2,.0)

    layer.weights = torch.tensor([
        [0.2,-0.3],
        [0.4,0.1]
    ],device = device, dtype=torch.float64)

    layer.biases = torch.tensor([0.0,0.0],device = device, dtype=torch.float64)

    X = torch.tensor([
        [1.0,2.0],
        [3.0,4.0],
    ],device=device, dtype=torch.float64)

    grad_o = torch.tensor([
        [0.5,-0.2],
        [0.1,0.3]
    ],device = device, dtype=torch.float64)

    layer.forward(X)
    layer.backward(grad_o)

    analytical_grad = layer.grad_w.clone()

    epsilon = 1e-5
    numerical_grad = torch.zeros_like(layer.weights, device=device)

    for row in range(layer.weights.shape[0]):
        for col in range(layer.weights.shape[1]):
            original_value = layer.weights[row,col].item()

            layer.weights[row,col] = original_value + epsilon
            output_plus = layer.forward(X)
            loss_plus = torch.sum(output_plus * grad_o)

            layer.weights[row,col] = original_value - epsilon
            output_minus = layer.forward(X)
            loss_minus = torch.sum(output_minus * grad_o)

            numerical_grad[row,col] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            layer.weights[row,col] = original_value

    torch.testing.assert_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-5,
        atol=1e-7)

def test_linear_layer_bias_gradient_matches_numerical_gradient():
    layer = LinearLayer(
        input_dim=2,
        neurons=2,
        learning_rate=0.0,
    )

    layer.weights = torch.tensor([
        [0.2, -0.3],
        [0.4, 0.1],
    ], device = device, dtype=torch.float64)

    layer.biases = torch.tensor([0.1, -0.2], device = device, dtype=torch.float64)

    X = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ], device = device, dtype=torch.float64)

    grad_o = torch.tensor([
        [0.5, -0.2],
        [0.1, 0.3],
    ], device = device, dtype=torch.float64)

    layer.forward(X)
    layer.backward(grad_o)

    analytical_grad = layer.grad_b.clone()

    epsilon = 1e-5
    numerical_grad = torch.zeros_like(layer.biases, device=device)

    for index in range(layer.biases.shape[0]):
        original_value = layer.biases[index].item()

        layer.biases[index] = original_value + epsilon
        output_plus = layer.forward(X)
        loss_plus = torch.sum(output_plus * grad_o)

        layer.biases[index] = original_value - epsilon
        output_minus = layer.forward(X)
        loss_minus = torch.sum(output_minus * grad_o)

        numerical_grad[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        layer.biases[index] = original_value

    torch.testing.assert_close(
        analytical_grad,
        numerical_grad,
        rtol=1e-5,
        atol=1e-7,
    )
