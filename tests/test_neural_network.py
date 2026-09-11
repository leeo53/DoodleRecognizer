# test_neural_network.py

import torch

from network.nn import NeuralNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_linear_model():
    torch.manual_seed(0)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=1000,
    )

    model.add_linear_layer(2,4)
    model.add_ReLU_layer()
    model.add_linear_layer(4,2)
    model.add_output_softmax_layer()

    return model


def make_linear_dataset():
    X = torch.tensor([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.5, -1.0],
        [1.0, 1.5],
        [2.0, 1.0],
        [1.0, 2.0],
    ], device=device)

    y = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], device=device)

    return X, y

def make_cnn_model():
    torch.manual_seed(0)

    model = NeuralNetwork(
        learning_rate=0.05,
        epochs=500
    )

    model.add_convolution_layer(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        stride=1,
        padding=0
    )
    model.add_ReLU_layer()

    model.add_max_pooling_layer(
        kernel_size=2,
        stride=2
    )

    model.add_flatten_layer()

    model.add_linear_layer(
        input_dim=2 * 2 * 2,
        output_dim=2
    )

    model.add_output_softmax_layer()

    return model

def make_cnn_dataset():
    X = torch.tensor([
        # Class 0: vertical lines
        [[
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
        ]],

        [[
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ]],

        [[
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ]],

        # Class 1: horizontal lines
        [[
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]],

        [[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]],

        [[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ]],
    ], dtype=torch.float32, device=device)

    y = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], device=device)

    return X, y



def test_linear_forward_returns_one_probability_per_class():
    X, _ = make_linear_dataset()
    model = make_linear_model()

    probabilities = model.forward(X)

    assert probabilities.shape == (X.shape[0], model.output_size)

def test_cnn_forward_returns_one_probability_per_class():
    X, _ = make_cnn_dataset()
    model = make_cnn_model()
    probabilities = model.forward(X)
    assert probabilities.shape == (X.shape[0], model.output_size)


def test_linear_softmax_probabilities_sum_to_one():
    X, _ = make_linear_dataset()
    model = make_linear_model()

    probabilities = model.forward(X)
    row_sums = torch.sum(probabilities, dim=1)

    torch.testing.assert_close(
        row_sums,
        torch.ones(X.shape[0], device=device),
        atol=1e-7,
        rtol=1e-7
    )

def test_cnn_softmax_probabilities_sum_to_one():
    X, _ = make_cnn_dataset()
    model = make_cnn_model()

    probabilities = model.forward(X)
    row_sums = torch.sum(probabilities, dim=1)

    torch.testing.assert_close(
        row_sums,
        torch.ones(X.shape[0], device=device),
        atol=1e-7,
        rtol=1e-7
    )


def test_training_linear_reduces_loss():
    X, y = make_linear_dataset()
    model = make_linear_model()

    initial_probs = model.forward(X)
    initial_loss = model.cross_entropy_error(y, initial_probs)

    model.fit(X, y)

    final_probs = model.forward(X)
    final_loss = model.cross_entropy_error(y, final_probs)

    assert final_loss < initial_loss

def test_training_cnn_reduces_loss():
    X, y = make_cnn_dataset()
    model = make_cnn_model()

    initial_probs = model.forward(X)
    initial_loss = model.cross_entropy_error(y, initial_probs)

    model.fit(X, y)

    final_probs = model.forward(X)
    final_loss = model.cross_entropy_error(y, final_probs)

    assert final_loss < initial_loss

def test_linear_model_learns_simple_two_class_problem():
    X, y = make_linear_dataset()
    model = make_linear_model()

    model.fit(X, y)

    final_probs = model.forward(X)
    predictions = torch.argmax(final_probs, dim=1)
    targets = torch.argmax(y, dim=1)

    assert torch.equal(predictions, targets)

def test_cnn_model_learns_simple_two_class_problem():
    X, y = make_cnn_dataset()
    model = make_cnn_model()

    model.fit(X, y)

    final_probs = model.forward(X)
    predictions = torch.argmax(final_probs, dim=1)
    targets = torch.argmax(y, dim=1)

    assert torch.equal(predictions, targets)

def test_linear_save_load():
    X, _ = make_linear_dataset()

    original_model = make_linear_model()
    original_probs = original_model.forward(X)

    original_model.save("test_model.pkl")

    loaded_model = NeuralNetwork()
    loaded_model.load("test_model.pkl")

    loaded_probs = loaded_model.forward(X)

    torch.testing.assert_close(
        original_probs,
        loaded_probs
    )

def test_cnn_save_load():
    X, _ = make_cnn_dataset()

    original_model = make_cnn_model()
    original_probs = original_model.forward(X)

    original_model.save("test_model.pkl")

    loaded_model = NeuralNetwork()
    loaded_model.load("test_model.pkl")

    loaded_probs = loaded_model.forward(X)

    torch.testing.assert_close(
        original_probs,
        loaded_probs
    )
