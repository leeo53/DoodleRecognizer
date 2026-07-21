# test_neural_network.py

import numpy as np

from network.simple_nn import NeuralNetwork


def make_model():
    np.random.seed(0)

    model = NeuralNetwork(
        input_size=2,
        output_size=2,
        learning_rate=0.05,
        epochs=1000,
    )

    model.add_linear_layer(4)
    model.add_ReLU_layer()
    model.add_linear_layer(2)
    model.add_softmax_layer()

    return model


def make_dataset():
    X = np.array([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.5, -1.0],
        [1.0, 1.5],
        [2.0, 1.0],
        [1.0, 2.0],
    ])

    y = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ])

    return X, y


def test_forward_returns_one_probability_per_class():
    X, _ = make_dataset()
    model = make_model()

    probabilities = model.forward(X)

    assert probabilities.shape == (X.shape[0], model.output_size)


def test_softmax_probabilities_sum_to_one():
    X, _ = make_dataset()
    model = make_model()

    probabilities = model.forward(X)
    row_sums = np.sum(probabilities, axis=1)

    np.testing.assert_allclose(
        row_sums,
        np.ones(X.shape[0]),
        atol=1e-7,
    )


def test_training_reduces_loss():
    X, y = make_dataset()
    model = make_model()

    initial_probs = model.forward(X)
    initial_loss = model.cross_entropy_error(y, initial_probs)

    model.fit(X, y)

    final_probs = model.forward(X)
    final_loss = model.cross_entropy_error(y, final_probs)

    assert final_loss < initial_loss


def test_model_learns_simple_two_class_problem():
    X, y = make_dataset()
    model = make_model()

    model.fit(X, y)

    final_probs = model.forward(X)
    predictions = np.argmax(final_probs, axis=1)
    targets = np.argmax(y, axis=1)

    assert np.array_equal(predictions, targets)