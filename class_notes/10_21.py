import modern_robotics as mr
import numpy as np
import sympy as sym

# iiwa screw axes:

L1 = sym.Symbol('L1')
L2 = sym.Symbol('L2')
L3 = sym.Symbol('L3')
L4 = sym.Symbol('L4')
B1_s = [0,0,1,0,0,0]
B2_s = [1,0,0,0,L2,0]
B3_s = [0,0,1,0,0,0]
B4_s = [1,0,0,0,L3,0]
B5_s = [0,0,1,0,0,0]
B6_s = [1,0,0,0,L4,0]
B7_s = [0,0,1,0,0,0]

# body jacobian
J_B = sym.Matrix([B1_s, B2_s, B3_s, B4_s, B5_s, B6_s, B7_s]).T
J_B_subbed = J_B.subs({L1:1 , L2:2, L3:3, L4:4})
# find rank of body jacobian
print(J_B)
print(f"Rank of body jacobian: {J_B_subbed.rank()}")