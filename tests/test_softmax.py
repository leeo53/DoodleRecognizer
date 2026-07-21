import numpy as np

from network.simple_nn import Layer, Layer_Type

def test_softmax_rows_sum_to_one():
    softmax = Layer(3,3,Layer_Type.SOFTMAX)

    X= np.array([[1,2,3],
                [-1,0,1]]).astype(np.float32)

    result = softmax.forward(X)

    rows_sum = np.sum(result,axis=1)

    np.testing.assert_allclose(
        rows_sum,
        np.ones(2),
        atol=1e-7,
    )

def test_softmax_gives_equal_probs_for_equal_scores():
    softmax = Layer(3,3,Layer_Type.SOFTMAX)

    X = np.array([[2,2,2]]).astype(np.float32)

    result = softmax.forward(X)

    expected = np.array([[1/3,1/3,1/3]])

    np.testing.assert_allclose(result, expected, atol=1e-7)

def test_softmax_assigns_largest_prob_to_largest_score():
    softmax = Layer(3,3,Layer_Type.SOFTMAX)

    X = np.array([[1,5,2]]).astype(np.float32)

    result = softmax.forward(X)

    assert np.argmax(result,axis=1)[0] == 1

def test_softmax_handles_large_values():
    softmax = Layer(3,3,Layer_Type.SOFTMAX)

    X = np.array([[1000,1001,1002]]).astype(np.float32)

    result = softmax.forward(X)

    assert np.all(np.isfinite(result))

    np.testing.assert_allclose(np.sum(result,axis=1),[1.0])

def test_softmax_cross_entropy_backward_returns_expected_gradient():
    layer = Layer(2, 2, Layer_Type.SOFTMAX)

    logits = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    y = np.array([
        [1, 0],
        [0, 1],
    ])

    probabilities = layer.forward(logits)
    result = layer.backward(y)

    expected = (probabilities - y) / y.shape[0]

    np.testing.assert_allclose(result, expected)

