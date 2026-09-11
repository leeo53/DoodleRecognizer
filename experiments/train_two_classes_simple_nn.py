import numpy as np
import torch
from experiments.data_loader import ImageDataLoader

from network.nn import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50

# Reproducible NumPy shuffling
np.random.seed(42)

# Reproducible PyTorch weight initialization
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
data_loader = ImageDataLoader(
    128,
    ["apple.npy", "airplane.npy"],
    1000,
    "data",
    flatten=True
)
model = NeuralNetwork(
    data_loader=data_loader,
    learning_rate=0.01,
    epochs=epochs
)

model.add_linear_layer(data_loader.input_size,64)
model.add_ReLU_layer()

model.add_linear_layer(64,32)
model.add_ReLU_layer()

model.add_linear_layer(32,2)
model.add_output_softmax_layer()

model.run(False)