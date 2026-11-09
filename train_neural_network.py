
import numpy as np

# Training data: XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# Network structure: 2 inputs -> 4 hidden neurons -> 1 output
np.random.seed(42)

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 20000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("LEO - FIRST NEURAL NETWORK")
print("--------------------------")

for epoch in range(epochs):

    # Forward propagation
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # Calculate loss
    epsilon = 1e-8
    loss = -np.mean(
        Y * np.log(A2 + epsilon)
        + (1 - Y) * np.log(1 - A2 + epsilon)
    )

    # Backpropagation
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / len(X)
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (1 - A1 ** 2)

    dW1 = X.T @ dZ1 / len(X)
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update parameters
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss:.6f}")

# Test the trained network
print("\nLEO - XOR PREDICTIONS")
print("--------------------")

hidden = np.tanh(X @ W1 + b1)
predictions = sigmoid(hidden @ W2 + b2)

for i in range(len(X)):
    print(
        f"Input: {X[i]} | "
        f"Prediction: {predictions[i][0]:.4f} | "
        f"Expected: {Y[i][0]:.0f}"
    )
