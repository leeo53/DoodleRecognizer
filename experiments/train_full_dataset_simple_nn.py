import torch
import numpy as np
from experiments.data_loader import ImageDataLoader
from network.nn import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reproducible NumPy shuffling
np.random.seed(42)

# Reproducible PyTorch weight initialization
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data_loader = ImageDataLoader(
    batch_size = 128,
    classes = None,
    samples_per_class = 10000,
    data_folder = "data",
    flatten = True
)
model = NeuralNetwork(
    data_loader=data_loader,
    learning_rate=0.01,
    epochs=60,
    batch_size=128,
)

model.add_linear_layer(
    input_dim=data_loader.input_size,
    output_dim=512
)
model.add_ReLU_layer()

model.add_linear_layer(
    input_dim=512,
    output_dim=512
)
model.add_ReLU_layer()

model.add_linear_layer(
    input_dim=512,
    output_dim=256
)
model.add_ReLU_layer()

model.add_linear_layer(
    input_dim=256,
    output_dim=data_loader.num_classes
)
model.add_output_softmax_layer()

model.file_name = "doodle_simple_nn.pkl"

model.run(save=False)