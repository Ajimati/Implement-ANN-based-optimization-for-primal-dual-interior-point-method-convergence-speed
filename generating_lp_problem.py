# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For dataset storage
import cvxpy as cp  # Optimization modeling library

# Generate a random linear programming problem
def generate_lp_problem(n=20, m=10):
    np.random.seed()  # Ensure randomness for each call
    A = np.random.randn(m, n)  # Generate a random constraint matrix A (m constraints, n variables)
    b = np.random.rand(m) * 10  # Generate positive constraint bounds vector b
    c = np.random.rand(n) * 10  # Generate objective coefficients vector c
    return A, b, c  # Return problem components
