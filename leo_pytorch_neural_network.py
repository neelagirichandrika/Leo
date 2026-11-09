
import torch
import torch.nn as nn

torch.manual_seed(42)

print("LEO - PYTORCH NEURAL NETWORK")
print("--------------------------------")

# XOR training data
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# Build neural network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Loss and optimiser
loss_function = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.03
)

# Training
epochs = 10000

for epoch in range(epochs):

    # Forward propagation
    predictions = model(X)

    # Calculate loss
    loss = loss_function(predictions, Y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        print(
            f"Epoch: {epoch + 1}, "
            f"Loss: {loss.item():.6f}"
        )

# Test the network
print("\nLEO - XOR PREDICTIONS")
print("--------------------------------")

with torch.no_grad():

    predictions = model(X)

    for i in range(len(X)):
        print(
            f"Input: {X[i].tolist()} | "
            f"Prediction: {predictions[i].item():.4f} | "
            f"Expected: {Y[i].item():.0f}"
        )

print("\nTraining completed!")
