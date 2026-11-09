
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(7)

print("LEO - VALIDATION EXPERIMENT")
print("---------------------------")

# Generate dataset
X = np.linspace(-3, 3, 120).reshape(-1, 1)
Y = np.sin(1.5 * X[:, 0]) + np.random.normal(0, 0.15, 120)

# Shuffle and split: 70% training, 15% validation, 15% testing
indices = np.random.permutation(len(X))

train_idx = indices[:84]
val_idx = indices[84:102]
test_idx = indices[102:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# Select regularisation using validation data
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
best_alpha = None
best_val_loss = float("inf")

for alpha in alphas:

    model = make_pipeline(
        PolynomialFeatures(degree=10),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    model.fit(X_train, Y_train)

    val_predictions = model.predict(X_val)
    val_loss = mean_squared_error(Y_val, val_predictions)

    print(f"Alpha: {alpha:7} | Validation loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_alpha = alpha

print("\nBEST MODEL SETTINGS")
print("Selected alpha:", best_alpha)
print("Best validation loss:", best_val_loss)

# Train final model using training + validation data
X_final = np.vstack((X_train, X_val))
Y_final = np.concatenate((Y_train, Y_val))

final_model = make_pipeline(
    PolynomialFeatures(degree=10),
    StandardScaler(),
    Ridge(alpha=best_alpha)
)

final_model.fit(X_final, Y_final)

# Evaluate once on untouched test data
test_predictions = final_model.predict(X_test)
test_loss = mean_squared_error(Y_test, test_predictions)

print("\nFINAL TEST RESULT")
print("Test loss:", test_loss)
print("\nExperiment completed!")
