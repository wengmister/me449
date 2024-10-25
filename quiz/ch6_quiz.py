import numpy as np
import modern_robotics as mr

# Newton-raphson method to find root of a function:
def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=2):
    x = x0
    for i in range(max_iter):
        x = x - f(x)/f_prime(x)
        if abs(f(x)) < tol:
            return x
    return x

# Q1
def f1(x):
    return x**2 - 9

def f1_prime(x):
    return 2*x

def f2(x):
    return x**2 - 4

def f2_prime(x):
    return 2*x

x1 = newton_raphson(f1, f1_prime, 1)
x2 = newton_raphson(f2, f2_prime, 1)
print(f"Result for Q1: {x1}, {x2}")


# Q2
J1_B = np.array([0,0,1,0,3,0])
J2_B = np.array([0,0,1,0,2,0])
J3_B = np.array([0,0,1,0,1,0])

Blist = np.column_stack([J1_B, J2_B, J3_B])
print(f"Blist: {Blist}")

T_sb = np.array([[1, 0, 0, 3],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
T_sd = np.array([[-0.585, -0.811, 0, 0.076],[0.811, -0.585, 0, 2.608],[0,0,1,0],[0,0,0,1]])
result = mr.IKinBody(Blist, T_sb, T_sd, np.array([np.pi/4, np.pi/4, np.pi/4]), 0.01, 0.001)

print(f"Result for Q2: {result}")