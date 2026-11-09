
import numpy as np

np.random.seed(7)

print("LEO - UNDERFITTING AND OVERFITTING")
print("-----------------------------------")

# Generate noisy data
X = np.linspace(-3, 3, 120)
Y = np.sin(1.5 * X) + np.random.normal(0, 0.15, 120)

# Separate training and testing examples
indices = np.random.permutation(len(X))

train_indices = indices[:12]
test_indices = indices[12:]

X_train = X[train_indices]
Y_train = Y[train_indices]

X_test = X[test_indices]
Y_test = Y[test_indices]

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# Compare models with different complexity
for degree in [1, 3, 10]:

    # Fit a polynomial model
    weights = np.polyfit(X_train, Y_train, degree)

    # Predictions
    train_predictions = np.polyval(weights, X_train)
    test_predictions = np.polyval(weights, X_test)

    # Calculate errors
    train_loss = np.mean((train_predictions - Y_train) ** 2)
    test_loss = np.mean((test_predictions - Y_test) ** 2)

    print("\nPolynomial degree:", degree)
    print("Training loss:", round(train_loss, 6))
    print("Testing loss:", round(test_loss, 6))

print("\nExperiment completed!")
