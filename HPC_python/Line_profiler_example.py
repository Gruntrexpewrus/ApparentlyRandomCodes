# Created by LP
# Date: 2025-03-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

#kernprof -l -v ~/Line_profiler_example.py

import cProfile
import numpy as np
import os

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_deriv(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# 🧠 Complex function: NN training
@profile
def train_numpy_nn():
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)

    W1 = np.random.randn(10, 64) * 0.1
    b1 = np.zeros((1, 64))
    W2 = np.random.randn(64, 1) * 0.1
    b2 = np.zeros((1, 1))

    learning_rate = 0.01

    for epoch in range(10):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2

        loss = mse(y_pred, y)

        grad_y_pred = mse_deriv(y_pred, y)
        grad_W2 = a1.T @ grad_y_pred
        grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)

        grad_a1 = grad_y_pred @ W2.T
        grad_z1 = grad_a1 * relu_deriv(z1)
        grad_W1 = X.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        W2 -= learning_rate * grad_W2
        b2 -= learning_rate * grad_b2

# 🧮 Simple function
@profile
def simple_math_loop():
    total = 0
    for i in range(100000):
        total += i ** 0.5
    return total

# 🔢 More complex function
@profile
def matrix_chain_multiplication():
    np.random.seed(42)
    A = np.random.rand(100, 200)
    B = np.random.rand(200, 300)
    C = np.random.rand(300, 400)
    D = np.random.rand(400, 500)
    result = A @ B @ C @ D
    return result

# 🏁 Main script
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "profiling_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "line_profiler.prof")

    profiler = cProfile.Profile()
    profiler.enable()

    simple_math_loop()
    matrix_chain_multiplication()
    train_numpy_nn()

    profiler.disable()
    profiler.dump_stats(output_path)
    print(f"cProfile output saved to: {output_path}")

if __name__ == "__main__":
    main()
    
    
    """
    Wrote profile results to Line_profiler_example.py.lprof
Timer unit: 1e-06 s

Total time: 0.010538 s
File: /Users/leonardoplacidi/Desktop/LOCALDEVELOPMENT/ApparentlyRandomCodes/HPC_python/Line_profiler_example.py
Function: train_numpy_nn at line 19

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    19                                           @profile
    20                                           def train_numpy_nn():
    21         1         19.0     19.0      0.2      np.random.seed(42)
    22         1        422.0    422.0      4.0      X = np.random.randn(1000, 10)
    23         1         41.0     41.0      0.4      y = np.random.randn(1000, 1)
    24                                           
    25         1         41.0     41.0      0.4      W1 = np.random.randn(10, 64) * 0.1
    26         1          7.0      7.0      0.1      b1 = np.zeros((1, 64))
    27         1          8.0      8.0      0.1      W2 = np.random.randn(64, 1) * 0.1
    28         1          1.0      1.0      0.0      b2 = np.zeros((1, 1))
    29                                           
    30         1          0.0      0.0      0.0      learning_rate = 0.01
    31                                           
    32        11          5.0      0.5      0.0      for epoch in range(10):
    33        10       3806.0    380.6     36.1          z1 = X @ W1 + b1
    34        10        893.0     89.3      8.5          a1 = relu(z1)
    35        10        269.0     26.9      2.6          z2 = a1 @ W2 + b2
    36        10          5.0      0.5      0.0          y_pred = z2
    37                                           
    38        10        315.0     31.5      3.0          loss = mse(y_pred, y)
    39                                           
    40        10         84.0      8.4      0.8          grad_y_pred = mse_deriv(y_pred, y)
    41        10        269.0     26.9      2.6          grad_W2 = a1.T @ grad_y_pred
    42        10        132.0     13.2      1.3          grad_b2 = np.sum(grad_y_pred, axis=0, keepdims=True)
    43                                           
    44        10       1366.0    136.6     13.0          grad_a1 = grad_y_pred @ W2.T
    45        10       1471.0    147.1     14.0          grad_z1 = grad_a1 * relu_deriv(z1)
    46        10        839.0     83.9      8.0          grad_W1 = X.T @ grad_z1
    47        10        388.0     38.8      3.7          grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)
    48                                           
    49        10         62.0      6.2      0.6          W1 -= learning_rate * grad_W1
    50        10         27.0      2.7      0.3          b1 -= learning_rate * grad_b1
    51        10         22.0      2.2      0.2          W2 -= learning_rate * grad_W2
    52        10         46.0      4.6      0.4          b2 -= learning_rate * grad_b2

Total time: 0.053825 s
File: /Users/leonardoplacidi/Desktop/LOCALDEVELOPMENT/ApparentlyRandomCodes/HPC_python/Line_profiler_example.py
Function: simple_math_loop at line 55

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    55                                           @profile
    56                                           def simple_math_loop():
    57         1          1.0      1.0      0.0      total = 0
    58    100001      21627.0      0.2     40.2      for i in range(100000):
    59    100000      32197.0      0.3     59.8          total += i ** 0.5
    60         1          0.0      0.0      0.0      return total

Total time: 0.00715 s
File: /Users/leonardoplacidi/Desktop/LOCALDEVELOPMENT/ApparentlyRandomCodes/HPC_python/Line_profiler_example.py
Function: matrix_chain_multiplication at line 63

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    63                                           @profile
    64                                           def matrix_chain_multiplication():
    65         1         35.0     35.0      0.5      np.random.seed(42)
    66         1        226.0    226.0      3.2      A = np.random.rand(100, 200)
    67         1        533.0    533.0      7.5      B = np.random.rand(200, 300)
    68         1       1051.0   1051.0     14.7      C = np.random.rand(300, 400)
    69         1       1739.0   1739.0     24.3      D = np.random.rand(400, 500)
    70         1       3565.0   3565.0     49.9      result = A @ B @ C @ D
    71         1          1.0      1.0      0.0      return result
    """