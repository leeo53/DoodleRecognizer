import numpy as np

from network.simple_nn import NeuralNetwork

X = np.array([
    [0.0,0.1,0.5],
    [9.9,9.7,8.0],
    [11.0,10.0,9.9],
    [1.0,0.1,1.8],
    [-1.3,0.0,1.2]
])

y = np.array([
    [0,1],
    [1,0],
    [1,0],
    [0,1],
    [0,1]
])

model = NeuralNetwork(
    input_size=3,
    output_size=2,
    learning_rate=0.05,
    epochs=1000
)

model.add_linear_layer(3)
model.add_ReLU_layer()
model.add_linear_layer(3)
model.add_ReLU_layer()
model.add_linear_layer(2)
model.add_softmax_layer()

model.fit(X, y)

final_probs = model.forward(X)
predictions = np.argmax(final_probs, axis=1)
targets = np.argmax(y, axis=1)

print("Probabilities:")
print(final_probs)

print("Predictions:", predictions)
print("Targets:    ", targets)
print("Accuracy:", np.mean(predictions == targets))

X_test = np.array([
    [0,0,0],
    [10,10,10],
])

y_test = np.array([
    [0,1],
    [1,0],
])

y_pred = model.forward(X_test)
predictions = np.argmax(y_pred, axis=1)
targets = np.argmax(y_test, axis=1)
print("Probabilities:")
print(y_pred)
print("Predictions:")
print(predictions)
print("Targets:")
print(targets)
print("Accuracy:", np.mean(predictions == targets))