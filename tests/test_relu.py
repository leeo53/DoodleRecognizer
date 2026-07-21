import numpy as np

from network.simple_nn import Layer, Layer_Type


def test_relu_sets_negative_values_to_zero():
    relu = Layer(3,3, Layer_Type.RELU, None)

    X = np.array([[1,2,-3],
                  [4,-5,-6],
                  [-7,8,-9]]).astype(np.float32)
    result = relu.forward(X)
    expected = np.array([[1,2,0],
                         [4,0,0],
                         [0,8,0]]).astype(np.float32)

    np.testing.assert_array_equal(result, expected)

def test_relu_preserves_input_shape():
    relu = Layer(3,3, Layer_Type.RELU, None)
    X = np.array([[1,2,-3]]).astype(np.float32)

    result = relu.forward(X)

    assert result.shape == X.shape

def test_relu_backward_blocks_gradient_for_negative_inputs():
    relu = Layer(3,3, Layer_Type.RELU, None)

    X = np.array([[1,2,-3]]).astype(np.float32)

    relu.forward(X)

    upstream_gradient = np.array([[10,20,30]]).astype(np.float32)

    result = relu.backward(upstream_gradient)
    expected = np.array([[10,20,0]]).astype(np.float32)

    np.testing.assert_array_equal(result, expected)