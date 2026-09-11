import torch

from network.activation_layer import OutputSoftmaxLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_softmax_rows_sum_to_one():
    softmax = OutputSoftmaxLayer()

    X= torch.tensor([[1,2,3],
                [-1,0,1]], dtype=torch.float32).to(device)

    result = softmax.forward(X)

    rows_sum = torch.sum(result,dim=1)

    torch.testing.assert_close(
        rows_sum,
        torch.ones(2, device=device),
        atol=1e-7,
        rtol=1e-7
    )

def test_softmax_gives_equal_probs_for_equal_scores():
    softmax = OutputSoftmaxLayer()

    X = torch.tensor([[2,2,2]], dtype=torch.float32).to(device)

    result = softmax.forward(X)

    expected = torch.tensor([[1/3,1/3,1/3]], dtype=torch.float32).to(device)

    torch.testing.assert_close(result, expected, atol=1e-7, rtol=1e-7)

def test_softmax_assigns_largest_prob_to_largest_score():
    softmax = OutputSoftmaxLayer()

    X = torch.tensor([[1,5,2]], dtype=torch.float32).to(device)

    result = softmax.forward(X)

    assert torch.argmax(result,dim=1).item() == 1

def test_softmax_handles_large_values():
    softmax = OutputSoftmaxLayer()

    X = torch.tensor([[1000,1001,1002]], dtype=torch.float32).to(device)

    result = softmax.forward(layer_input=X)

    assert torch.all(torch.isfinite(result))

    torch.testing.assert_close(torch.sum(result,dim=1),torch.tensor([1.0],device=device, dtype=result.dtype))

def test_softmax_cross_entropy_backward_returns_expected_gradient():
    layer = OutputSoftmaxLayer()

    logits = torch.tensor([
        [2.0, 1.0],
        [1.0, 3.0],
    ],dtype=torch.float32).to(device)

    y = torch.tensor([
        [1, 0],
        [0, 1],
    ], dtype=torch.float32).to(device)

    probabilities = layer.forward(logits)
    result = layer.backward(y)

    expected = (probabilities - y) / y.shape[0]

    torch.testing.assert_close(result, expected)

def test_softmax_handles_large_negative_values():
    softmax = OutputSoftmaxLayer()

    X = torch.tensor(
        [[-1000.0, -1001.0, -1002.0]],
        device=device
    )

    result = softmax.forward(X)

    assert torch.all(torch.isfinite(result)).item()

    torch.testing.assert_close(
        torch.sum(result, dim=1),
        torch.tensor([1.0], device=device)
    )

