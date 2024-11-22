import sympy as sym
import modern_robotics as mr
import numpy as np

# Define symbolic variable λ (lambda)
lam = sym.symbols('λ')

# Create the coefficient matrix A
A = sym.Matrix([
    [0, 1, 0],
    [0, 0, 1],
    [-3, -2, -1]
])

# Create 3x3 identity matrix
I = sym.Matrix.eye(3)

# Calculate λI - A
char_matrix = lam*I - A

print("Characteristic matrix:")
print(char_matrix)

# Calculate determinant
char_equation = char_matrix.det()

# Expand to get a cleaner form
char_equation = sym.expand(char_equation)

print("\nCharacteristic equation:")
print(f"Q2: {char_equation} = 0")

eigenvalues = sym.solve(char_equation, lam)
print("\nEigenvalues:")
for i in eigenvalues:
    print(i.simplify())