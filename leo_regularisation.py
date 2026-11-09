
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(7)

print("LEO - REGULARISATION EXPERIMENT")
print("--------------------------------")

# Generate data
X = np.linspace(-3, 3, 120).reshape(-1, 1)
Y = np.sin(1.5 * X[:, 0]) + np.random.normal(0, 0.15, 120)

# Split data
indices = np.random.permutation(len(X))
train_indices = indices[:12]
test_indices = indices[12:]

X_train = X[train_indices]
Y_train = Y[train_indices]

X_test = X[test_indices]
Y_test = Y[test_indices]

# Compare regularisation strengths
for alpha in [0, 0.01, 1, 100]:

    model = make_pipeline(
        PolynomialFeatures(degree=10),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    model.fit(X_train, Y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_loss = mean_squared_error(Y_train, train_predictions)
    test_loss = mean_squared_error(Y_test, test_predictions)

    print(f"\nRegularisation strength: {alpha}")
    print(f"Training loss: {train_loss:.6f}")
    print(f"Testing loss: {test_loss:.6f}")

print("\nExperiment completed!")
