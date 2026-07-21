import numpy as np
import pytest

from network.simple_nn import NeuralNetwork

def test_correct_confident_prediction_has_low_loss():
    y = np.array([
        [1,0],
        [0,1]
    ])

    predictions = np.array([
        [0.99,0.01],
        [0.01,0.99]
    ])

    model = NeuralNetwork(
        input_size=2,
        output_size=2,
        learning_rate=0.05,
        epochs=1
    )

    loss = model.cross_entropy_error(y, predictions)

    assert loss < 0.02

def test_good_prediction_has_low_loss():
    y = np.array([
        [1,0],
        [0,1]
    ])

    good_predictions = np.array([
        [0.9,0.1],
        [0.1,0.9]
    ])

    bad_predictions = np.array([
        [0.1,0.9],
        [0.9,0.1]
    ])
    model = NeuralNetwork(
        input_size=2,
        output_size=2,
        learning_rate=0.05,
        epochs=1
    )

    good_loss = model.cross_entropy_error(y, good_predictions)
    bad_loss = model.cross_entropy_error(y, bad_predictions)
    assert good_loss < bad_loss

def test_cross_entropy_error_returns_expected():
    y = np.array([
        [1,0]
    ])

    predictions = np.array([
        [0.8,0.2],
    ])

    model = NeuralNetwork(
        input_size=2,
        output_size=2,
        learning_rate=0.05,
        epochs=1
    )

    result = model.cross_entropy_error(y, predictions)
    expected = -np.log(0.8)
    assert result == pytest.approx(expected)