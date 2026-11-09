
import numpy as np

# LEO's first artificial neuron

inputs = np.array([1, 2, 3])

weights = np.array([0.2, 0.4, -0.5])

bias = 0.1

# Calculate the weighted sum
weighted_sum = np.dot(inputs, weights) + bias

print("LEO - First Artificial Neuron")
print("------------------------------")
print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)
print("Neuron output:", weighted_sum)
