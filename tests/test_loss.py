import torch
import pytest

from network.nn import NeuralNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_correct_confident_prediction_has_low_loss():
    y = torch.tensor([
        [1,0],
        [0,1]
    ], device=device,dtype=torch.float32)

    predictions = torch.tensor([
        [0.99,0.01],
        [0.01,0.99]
    ], device=device, dtype=torch.float32)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=1
    )

    loss = model.cross_entropy_error(y, predictions)

    assert loss < 0.02

def test_good_prediction_has_low_loss():
    y = torch.tensor([
        [1,0],
        [0,1]
    ], device=device,dtype=torch.float32)

    good_predictions = torch.tensor([
        [0.9,0.1],
        [0.1,0.9]
    ], device=device,dtype=torch.float32)

    bad_predictions = torch.tensor([
        [0.1,0.9],
        [0.9,0.1]
    ], device=device,dtype=torch.float32)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=1
    )

    good_loss = model.cross_entropy_error(y, good_predictions)
    bad_loss = model.cross_entropy_error(y, bad_predictions)
    assert good_loss < bad_loss

def test_cross_entropy_error_returns_expected():
    y = torch.tensor([
        [1,0]
    ], device=device,dtype=torch.float32)

    predictions = torch.tensor([
        [0.8,0.2],
    ], device=device,dtype=torch.float32)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=1
    )

    result = model.cross_entropy_error(y, predictions)
    expected = -torch.log(torch.tensor(0.8, device=device))
    torch.testing.assert_close(result, expected)

def test_cross_entropy_averages_loss_across_batch():
    y = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ], device=device)

    predictions = torch.tensor([
        [0.8, 0.2],
        [0.1, 0.9]
    ], device=device)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=1
    )

    result = model.cross_entropy_error(y, predictions)

    expected = (
        -torch.log(torch.tensor(0.8, device=device))
        -torch.log(torch.tensor(0.9, device=device))
    ) / 2

    torch.testing.assert_close(result, expected)