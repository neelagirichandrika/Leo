
import numpy as np

# Training examples
inputs = np.array([1, 2, 3, 4], dtype=float)
expected = np.array([2, 4, 6, 8], dtype=float)

# Initial weight
weight = 0.0

# Learning rate
learning_rate = 0.01

# Training iterations
epochs = 100

print("LEO - FIRST LEARNING EXPERIMENT")
print("--------------------------------")

for epoch in range(epochs):

    # 1. Make predictions
    predictions = weight * inputs

    # 2. Calculate the error
    loss = np.mean((predictions - expected) ** 2)

    # 3. Calculate the gradient
    gradient = np.mean(
        2 * inputs * (predictions - expected)
    )

    # 4. Update the weight
    weight = weight - learning_rate * gradient

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss:.6f}, "
            f"Weight: {weight:.6f}"
        )

print("--------------------------------")
print("Training completed!")
print("Learned weight:", weight)

# Test the trained neuron
test_input = 5
prediction = weight * test_input

print("Test input:", test_input)
print("LEO prediction:", prediction)
print("Expected answer:", 10)
