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
    batch_size=128,
    classes=None,
    samples_per_class=10000,
    data_folder = "data",
    flatten=False)

model = NeuralNetwork(
    data_loader=data_loader,
    learning_rate=0.01,
    epochs=60,
    batch_size=128,
)

model.add_convolution_layer(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)
model.add_ReLU_layer()

model.add_convolution_layer(
    in_channels=32,
    out_channels=32,
    kernel_size=3,
    stride=1,
    padding=1
)
model.add_ReLU_layer()

model.add_max_pooling_layer(
    kernel_size=2,
    stride=2
)

model.add_convolution_layer(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
model.add_ReLU_layer()

model.add_convolution_layer(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1
)
model.add_ReLU_layer()

model.add_max_pooling_layer(
    kernel_size=2,
    stride=2
)

model.add_convolution_layer(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1
)
model.add_ReLU_layer()

model.add_max_pooling_layer(
    kernel_size=2,
    stride=2
)

model.add_flatten_layer()

model.add_linear_layer(
    input_dim=128 * 3 * 3,
    output_dim=256
)
model.add_ReLU_layer()

model.add_linear_layer(
    input_dim=256,
    output_dim=data_loader.num_classes
)

model.add_output_softmax_layer()

model.file_name = "doodle_cnn.pkl"

model.run(save=False)