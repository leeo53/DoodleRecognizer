import torch
import pickle


from network.convolution_layer import ConvLayer
from network.flatten_layer import FlattenLayer
from network.linear_layer import LinearLayer
from network.activation_layer import ReLULayer, SigmoidLayer, OutputSoftmaxLayer, TanhLayer
from network.max_pooling_layer import MaxPoolLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork:
    def __init__(self,
                 data_loader=None,
                 learning_rate=0.01,
                 epochs=100):
        '''
        for the current implementation the softmax layer must always be added to the end of the model
        :param data_loader: you can make this None and run the model without a data loader
        :param learning_rate:
        :param epochs:
        '''
        self.layers = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_size = 0
        self.class_names=None
        self.file_name = "model.pkl"
        if data_loader is not None:
            self.data_loader = data_loader
            self.class_names = self.data_loader.classes

    def add_linear_layer(self, input_dim, output_dim):
        '''
        add a linear layer to the model as the next layer
        :param input_dim: the input dimension
        :param output_dim: the output dimension
        '''
        self.layers.append(LinearLayer(input_dim=input_dim,
                                       neurons=output_dim,
                                       learning_rate=self.learning_rate))
        self.output_size = output_dim

    def add_ReLU_layer(self):
        '''
        add a ReLU layer to the model as the next layer
        '''
        self.layers.append(ReLULayer())

    def add_sigmoid_layer(self):
        '''
        add a sigmoid layer to the model as the next layer
        '''
        self.layers.append(SigmoidLayer())

    def add_tanh_layer(self):
        '''
        add a tanh layer to the model as the next layer
        '''
        self.layers.append(TanhLayer())

    def add_output_softmax_layer(self):
        '''
        must be used as the last activation layer and nowhere else
        '''
        self.layers.append(OutputSoftmaxLayer())

    def add_convolution_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        '''
        adds a convolution layer to the model as the next layer
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size:
        :param stride:
        :param padding:
        '''
        self.layers.append(ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            learning_rate=self.learning_rate
        ))

    def add_max_pooling_layer(self, kernel_size, stride):
        '''
        add a max pooling layer to the model as the next layer
        :param kernel_size:
        :param stride:
        '''
        self.layers.append(MaxPoolLayer(kernel_size, stride))

    def add_flatten_layer(self):
        '''
        add a flatten layer to the model as the next layer
        '''
        self.layers.append(FlattenLayer())

    def forward(self, input):
        '''
        do one forward pass through the entire model
        :param input: the input to the model
        :return: the output of the model as probabilities
        '''
        z=input
        for i,layer in enumerate(self.layers):
            z= layer.forward(z)
        return z

    def backward(self, targets):
        '''
        do one backward pass through the entire model updating the parameters
        :param targets: the labels for the data
        '''
        grad = targets
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit_data_loader(self):
        '''
        fit the model to the data from the data loader, can only be used if the data loader is not none
        :return: 0 if successful, else -1
        '''
        if len(self.layers) == 0 or self.data_loader is None:
            return -1

        epoch_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total=0
        test_accuracy = 0

        for epoch in range(self.epochs):
            self.data_loader.shuffle()
            epoch_loss = torch.tensor(0.0, device=device)
            correct = torch.tensor(0, device=device)
            total = 0

            batch_num = 0

            while (True):
                batch_X, batch_y = self.data_loader.next_batch(True)
                if batch_X is None:
                    break
                probs = self.forward(batch_X)
                loss = self.cross_entropy_error(batch_y, probs)
                epoch_loss += loss.item() * len(batch_X)

                predicted_classes = torch.argmax(probs, dim=1)
                true_classes = torch.argmax(batch_y, dim=1)

                correct += (predicted_classes == true_classes).sum()
                total += batch_y.shape[0]

                self.backward(batch_y)

                batch_num += 1

                if batch_num % 5000 == 0:
                    print(
                        f"Epoch {epoch}: batch {batch_num}",
                        flush=True
                    )

            epoch_loss = epoch_loss.item() / len(self.data_loader.train_samples)
            train_accuracy = correct.item() / total * 100
            if epoch % 10 == 0:
                test_loss, test_accuracy = self.evaluate()
                print(
                    f"Epoch {epoch}: "
                    f"loss = {epoch_loss:.6f}, "
                    f"training accuracy = {train_accuracy:.4f}%, "
                    f"testing accuracy = {test_accuracy*100:.4f}%"
                )
        train_accuracy = correct.item() / total * 100
        print(
            f"Epoch {self.epochs}: "
            f"loss = {epoch_loss:.6f}, "
            f"training accuracy = {train_accuracy:.4f}%, "
            f"testing accuracy = {test_accuracy*100:.4f}%"
        )
        return 0

    def fit(self, X, y):
        '''
        fits the model to the data X and y
        :param X: input data
        :param y: target values
        :return: 0 if successful, else -1
        '''
        if len(self.layers) == 0:
            return -1
        probs = self.forward(X)
        loss = self.cross_entropy_error(y, probs)
        for epoch in range(self.epochs):
            probs = self.forward(X)

            loss = self.cross_entropy_error(y, probs)

            predicted_classes = torch.argmax(probs, dim=1)
            true_classes = torch.argmax(y, dim=1)

            correct = (predicted_classes == true_classes).sum().item()
            accuracy = correct / y.shape[0] * 100

            self.backward(y)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"loss = {loss.item():.6f}, "
                    f"accuracy = {accuracy:.4f}%"
                )

        predicted_classes = torch.argmax(probs, dim=1)
        true_classes = torch.argmax(y, dim=1)

        correct = (predicted_classes == true_classes).sum().item()
        accuracy = correct / y.shape[0] * 100
        print(
            f"Epoch {self.epochs}: "
            f"loss = {loss.item():.6f}, "
            f"accuracy = {accuracy:.4f}%"
        )

        return 0

    def cross_entropy_error(self, y, probs):
        '''
        returns the cross entropy error given the target values and the probabilities
        :param y: target values
        :param probs: probabilities returned by the model
        :return: the cross entropy error
        '''
        probs = torch.clamp(probs, 1e-15, 1.0)
        return -torch.sum(y * torch.log(probs))/y.shape[0]

    def run(self, save):
        '''
        fits the model to the data supplied by the data loader, and evaluates it against the testing data,
        and saves it if save is True
        :param save: saves the model to the file denoted by self.file_name if True
        '''
        if self.data_loader is None:
            raise ValueError("Cannot run model because data_loader is None.")

        self.fit_data_loader()

        test_loss, test_accuracy = self.evaluate()
        print(f"Testing loss: {test_loss:.6f}")
        print(f"Testing accuracy: {test_accuracy * 100:.4f}")

        if save:
            self.save(self.file_name)

    def predict(self, input, k=5):
        '''
        sends the input through one forward pass and finds the top k predictions
        :param input: input data
        :param k: number of top predictions to return
        :return: the names of the top k predictions, the probabilities of the top k predictions
        '''
        probabilities = self.forward(input)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k,
            dim=1
        )

        predicted_names = [
            [self.class_names[index] for index in row]
            for row in top_indices.tolist()
        ]

        return predicted_names, top_probabilities

    def evaluate(self):
        '''
        uses the dataloader to check how the model performs on the testing data
        :return: the loss with respect to the testing data, the accuracy on the testing data
        '''
        correct = torch.tensor(0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        total = 0

        while True:
            batch_X, batch_y = self.data_loader.next_batch(False)

            if batch_X is None:
                break

            probs = self.forward(batch_X)
            loss = self.cross_entropy_error(batch_y, probs)

            predicted_classes = torch.argmax(probs, dim=1)
            true_classes = torch.argmax(batch_y, dim=1)

            correct += (predicted_classes == true_classes).sum()
            total += batch_y.shape[0]
            total_loss += loss * batch_y.shape[0]

        test_accuracy = correct.item() / total
        test_loss = total_loss.item() / total

        return test_loss, test_accuracy

    def save(self, filename):
        '''
        saves the model in a pickle file named filename
        :param filename: name of the file
        '''
        model_data = {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "layers": [],
            "class_names": self.class_names
        }
        for layer in self.layers:
            layer_type = type(layer).__name__

            layer_data = {
                "layer_type": layer_type
            }

            if isinstance(layer, LinearLayer):
                layer_data["input_dim"] = layer.input_dim
                layer_data["neurons"] = layer.neurons
                layer_data["weights"] = layer.weights.detach().cpu()
                layer_data["biases"] = layer.biases.detach().cpu()
            elif isinstance(layer, ConvLayer):
                layer_data["in_channels"] = layer.in_channels
                layer_data["out_channels"] = layer.out_channels
                layer_data["kernel_size"] = layer.kernel_size
                layer_data["stride"] = layer.stride
                layer_data["padding"] = layer.padding
                layer_data["kernels"] = layer.kernels.detach().cpu()
                layer_data["biases"] = layer.biases.detach().cpu()
            elif isinstance(layer, MaxPoolLayer):
                layer_data["kernel_size"] = layer.kernel_size
                layer_data["stride"] = layer.stride
            elif isinstance(layer, (ReLULayer, SigmoidLayer, TanhLayer, OutputSoftmaxLayer, FlattenLayer)):
                pass
            else:
                raise TypeError(f"Unsupported layer type: {layer_type}")

            model_data["layers"].append(layer_data)

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(model_data, file)

    def load(self, filename):
        '''
        loads the model from a pickle file named filename
        :param filename: name of the file
        '''
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "rb") as file:
            model_data = pickle.load(file)
        self.epochs = model_data["epochs"]
        self.learning_rate = model_data["learning_rate"]
        self.layers = []
        self.class_names = model_data["class_names"]
        for saved_layer in model_data["layers"]:
            layer_type = saved_layer["layer_type"]
            if layer_type == "LinearLayer":
                self.add_linear_layer(
                                saved_layer["input_dim"],
                                saved_layer["neurons"])

                self.layers[-1].weights = saved_layer["weights"].to(device)
                self.layers[-1].biases = saved_layer["biases"].to(device)
            elif layer_type == "ConvLayer":
                self.add_convolution_layer(
                    saved_layer["in_channels"],
                    saved_layer["out_channels"],
                    saved_layer["kernel_size"],
                    saved_layer["stride"],
                    saved_layer["padding"]
                )
                self.layers[-1].kernels = saved_layer["kernels"].to(device)
                self.layers[-1].biases = saved_layer["biases"].to(device)
            elif layer_type == "MaxPoolLayer":
                self.add_max_pooling_layer(
                    saved_layer["kernel_size"],
                    saved_layer["stride"]
                )
            elif layer_type == "ReLULayer":
                self.add_ReLU_layer()
            elif layer_type == "SigmoidLayer":
                self.add_sigmoid_layer()
            elif layer_type == "TanhLayer":
                self.add_tanh_layer()
            elif layer_type == "OutputSoftmaxLayer":
                self.add_output_softmax_layer()
            elif layer_type == "FlattenLayer":
                self.add_flatten_layer()
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")