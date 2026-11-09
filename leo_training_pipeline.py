
import numpy as np

np.random.seed(42)

print("LEO - TRAINING AND TESTING PIPELINE")
print("-----------------------------------")

# 1. Generate a dataset
X = np.random.uniform(-5, 5, (200, 1))
y = 3 * X + 2 + np.random.normal(0, 0.5, (200, 1))

# 2. Split into training and testing data
indices = np.random.permutation(len(X))

train_indices = indices[:160]
test_indices = indices[160:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# 3. Initialize model parameters
weight = 0.0
bias = 0.0

learning_rate = 0.01
epochs = 1000

# 4. Train the model
for epoch in range(epochs):

    predictions = X_train * weight + bias
    error = predictions - y_train

    weight_gradient = 2 * np.mean(error * X_train)
    bias_gradient = 2 * np.mean(error)

    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    if (epoch + 1) % 200 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch + 1}, Training Loss: {loss:.4f}")

# 5. Evaluate on unseen test data
test_predictions = X_test * weight + bias
test_loss = np.mean((test_predictions - y_test) ** 2)

print("\nRESULTS")
print("-----------------------------------")
print("Learned weight:", weight)
print("Learned bias:", bias)
print("Test loss:", test_loss)

print("\nExpected weight: 3")
print("Expected bias: 2")
