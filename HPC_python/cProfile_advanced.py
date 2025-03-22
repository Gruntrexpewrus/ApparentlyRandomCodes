# Created by LP
# Date: 2025-03-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

import cProfile
import numpy as np
import os

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Mean Squared Error loss and derivative
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_deriv(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# Neural Network training function
def train_numpy_nn():
    np.random.seed(42)

    # Dataset: 1000 samples, 10 features
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)

    # Initialize weights
    W1 = np.random.randn(10, 64) * 0.1
    b1 = np.zeros((1, 64))
    W2 = np.random.randn(64, 1) * 0.1
    b2 = np.zeros((1, 1))

    learning_rate = 0.01

    for epoch in range(10):  # Small number of epochs for quick profiling
        # Forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2

        # Loss
        loss = mse(y_pred, y)
        # print(f"Epoch {epoch+1}, Loss: {loss:.4f}")  # Optional

        # Backward pass
        grad_y_pred = mse_deriv(y_pred, y)         # (1000, 1)
        grad_W2 = a1.T @ grad_y_pred               # (64, 1)
        grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)

        grad_a1 = grad_y_pred @ W2.T               # (1000, 64)
        grad_z1 = grad_a1 * relu_deriv(z1)         # (1000, 64)
        grad_W1 = X.T @ grad_z1                    # (10, 64)
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        # Update weights
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

def main():
    # Get the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the folder inside that directory
    output_dir = os.path.join(current_dir, "profiling_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "profile_numpy_nn.prof")
    
    profiler = cProfile.Profile()
    profiler.enable()

    train_numpy_nn()

    profiler.disable()
    
    profiler.dump_stats(output_path)
    print(f"Profiling complete. Output saved to {output_path}.")

if __name__ == "__main__":
    main()