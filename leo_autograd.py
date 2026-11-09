
import torch

print("LEO - AUTOMATIC DIFFERENTIATION")
print("--------------------------------")

# Create a parameter that PyTorch must learn
weight = torch.tensor(2.0, requires_grad=True)

# Input
x = torch.tensor(3.0)

# Expected answer
target = torch.tensor(10.0)

# Forward calculation
prediction = weight * x

# Calculate error
loss = (prediction - target) ** 2

print("Initial weight:", weight.item())
print("Prediction:", prediction.item())
print("Loss:", loss.item())

# Automatically calculate gradient
loss.backward()

print("Gradient:", weight.grad.item())

# Update weight
learning_rate = 0.1

with torch.no_grad():
    weight -= learning_rate * weight.grad

# Clear old gradient
weight.grad.zero_()

print("\nAfter one learning step:")
print("Updated weight:", weight.item())

print("\nExperiment completed!")
