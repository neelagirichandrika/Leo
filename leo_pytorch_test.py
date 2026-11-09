import torch

print("LEO - PYTORCH TEST")
print("------------------")

print("PyTorch version:", torch.__version__)

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

print("First tensor:", x)

# Perform a calculation
y = x * 2

print("After multiplication:", y)

# Check available devices
print("CUDA available:", torch.cuda.is_available())

print("CPU available: True")

print("\nLEO PyTorch test completed!")