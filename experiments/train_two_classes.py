import os
import numpy as np

from network.simple_nn import NeuralNetwork
epochs = 700
np.random.seed(42)

apple_class = np.load("data/apple.npy")
airplane_class = np.load("data/airplane.npy")

apple_class = apple_class[:1000]
airplane_class = airplane_class[:1000]

X = np.concatenate((apple_class, airplane_class), axis=0)
apple_labels = np.tile([1,0],(apple_class.shape[0],1))
airplane_labels = np.tile([0,1],(airplane_class.shape[0],1))
y = np.vstack((apple_labels,airplane_labels))

X = X.astype(np.float32) / 255.0
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

X_train = X[:1600]
y_train = y[:1600]
X_test = X[1600:]
y_test = y[1600:]

model = NeuralNetwork(X_train.shape[1], y_train.shape[1], learning_rate=0.01, epochs=epochs)

model.add_linear_layer(64)
model.add_ReLU_layer()

model.add_linear_layer(32)
model.add_ReLU_layer()

model.add_linear_layer(2)
model.add_softmax_layer()

model.fit(X_train, y_train)

final_probs = model.forward(X)
predictions = np.argmax(final_probs, axis=1)
targets = np.argmax(y, axis=1)
print("Model Training Accuracy:", np.mean(predictions == targets))

test_probs = model.forward(X_test)
predictions = np.argmax(test_probs, axis=1)
targets = np.argmax(y_test, axis=1)
print("Model Testing Accuracy:", np.mean(predictions == targets))