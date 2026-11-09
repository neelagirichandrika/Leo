
import numpy as np

# Training data
X = np.array([
    [1, 2, 3],
    [2, 1, 4],
    [3, 2, 1],
    [4, 3, 2],
    [2, 4, 3],
    [5, 1, 2]
], dtype=float)

# Expected answers
y = 2*X[:, 0] + 3*X[:, 1] - 1.5*X[:, 2] + 4

# Initialize weights and bias
np.random.seed(42)

weights = np.random.randn(3)
bias = 0.0

learning_rate = 0.04
epochs = 3000

print("LEO - MULTIPLE INPUT NEURON")
print("---------------------------")

# Training
for epoch in range(epochs):

    # Forward propagation
    predictions = X @ weights + bias

    # Calculate loss
    loss = np.mean((predictions - y) ** 2)

    # Calculate gradients
    error = predictions - y

    weight_gradients = (2 / len(X)) * (X.T @ error)
    bias_gradient = 2 * np.mean(error)

    # Update parameters
    weights -= learning_rate * weight_gradients
    bias -= learning_rate * bias_gradient

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss:.8f}"
        )

print("---------------------------")
print("Training completed!")

print("Learned weights:", weights)
print("Learned bias:", bias)

print("\nExpected weights: [2, 3, -1.5]")
print("Expected bias: 4")
