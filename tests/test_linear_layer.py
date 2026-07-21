import numpy as np

from network.simple_nn import Layer,Layer_Type

def test_linear_layer_forward_calculates_expected():
    layer = Layer(2,2,Layer_Type.LINEAR,.05)

    layer.weights = np.array([
        [3.0,4.0],
        [5.0,6.0],
    ])

    layer.biases = np.array([1.0,2.0])

    X = np.array([[1.0,2.0]])

    result = layer.forward(X)

    expected = np.array([[14.0,18.0]])

    np.testing.assert_allclose(result, expected)

def test_linear_layer_returns_correct_shape():
    layer = Layer(3,4,Layer_Type.LINEAR,.05)
    X = np.ones((5,3))
    result = layer.forward(X)
    assert result.shape == (5,4)

def test_linear_layer_weight_gradient_matches_numerical_gradient():
    layer = Layer(2,2,Layer_Type.LINEAR,.0)

    layer.weights = np.array([
        [0.2,-0.3],
        [0.4,0.1]
    ])

    layer.biases = np.array([0.0,0.0])

    X = np.array([
        [1.0,2.0],
        [3.0,4.0],
    ])

    grad_o = np.array([
        [0.5,-0.2],
        [0.1,0.3]
    ])

    layer.forward(X)
    layer.backward(grad_o)

    analytical_grad = layer.grad_w.copy()

    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.weights)

    for row in range(layer.weights.shape[0]):
        for col in range(layer.weights.shape[1]):
            original_value = layer.weights[row,col]

            layer.weights[row,col] = original_value + epsilon
            output_plus = layer.forward(X)
            loss_plus = np.sum(output_plus * grad_o)

            layer.weights[row,col] = original_value - epsilon
            output_minus = layer.forward(X)
            loss_minus = np.sum(output_minus * grad_o)

            numerical_grad[row,col] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            layer.weights[row,col] = original_value

    np.testing.assert_allclose(
        analytical_grad,
        numerical_grad,
        rtol=1e-5,
        atol=1e-7)

def test_linear_layer_bias_gradient_matches_numerical_gradient():
    layer = Layer(
        input_dim=2,
        neurons=2,
        layer_type=Layer_Type.LINEAR,
        learning_rate=0.0,
    )

    layer.weights = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    layer.biases = np.array([0.1, -0.2])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    grad_o = np.array([
        [0.5, -0.2],
        [0.1, 0.3],
    ])

    layer.forward(X)
    layer.backward(grad_o)

    analytical_grad = layer.grad_b.copy()

    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.biases)

    for index in range(layer.biases.shape[0]):
        original_value = layer.biases[index]

        layer.biases[index] = original_value + epsilon
        output_plus = layer.forward(X)
        loss_plus = np.sum(output_plus * grad_o)

        layer.biases[index] = original_value - epsilon
        output_minus = layer.forward(X)
        loss_minus = np.sum(output_minus * grad_o)

        numerical_grad[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        layer.biases[index] = original_value

    np.testing.assert_allclose(
        analytical_grad,
        numerical_grad,
        rtol=1e-5,
        atol=1e-7,
    )
