import modern_robotics as mr
import numpy as np

# Set NumPy print options to limit float precision to 2 decimal places
np.set_printoptions(precision=2, suppress=True)

T_sa = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,1],[0,0,0,1]])
T_as = mr.TransInv(T_sa)
print(T_as)

T_sb = np.array([[1,0,0,0],[0,0,1,2],[0,-1,0,0],[0,0,0,1]])
T_bs = mr.TransInv(T_sb)
print(f"Result for P2: {T_bs}")

T_ab = np.dot(T_as, T_sb)
print(f"Result for q3: {T_ab}")

V_b = np.array([1,2,3,1]).T
V_s = np.dot(T_sb, V_b)
print(f"Result for p5: {V_s}")

T_as_adj = mr.Adjoint(T_as)
print(T_as_adj)
Twist_s = np.array([3,2,1,-1,-2,-3]).T
Twist_a = np.dot(T_as_adj, Twist_s)
print(f"Result for p7: {Twist_a}")

T_sa_log = mr.MatrixLog6(T_sa)
print(T_sa_log)


# P8
T_sa_log = mr.MatrixLog6(T_sa)
T_sa_vec = mr.se3ToVec(T_sa_log)

T_sa_ang = mr.AxisAng6(T_sa_vec)
print(f"result for P8: {T_sa_ang}")
# P9
P9_vec = np.array([0,1,2,3,0,0]).T
P9_se3 = mr.VecTose3(P9_vec)
P9_T = mr.MatrixExp6(P9_se3)
print(f"result for P9: \n{P9_T}")

# P10
T_bs_adj = mr.Adjoint(T_bs)
F_b = np.array([1,0,0,2,1,0]).T
F_s = np.dot(T_bs_adj.T, F_b)
print(f"result for P10: {F_s}")

# P11
T_11_orig = np.array([[0,-1,0,3],[1,0,0,0],[0,0,1,1,],[0,0,0,1]])
T_11_inv = mr.TransInv(T_11_orig)
print(f"result for P11: \n {T_11_inv}")

# P12
V_12 = np.array([1,0,0,0,2,3]).T
T_12 = mr.VecTose3(V_12)
print(f"Result for P12: \n {T_12}")

# P13
s_13 = mr.ScrewToAxis(np.array([0,0,2]),np.array([1,0,0]),1)
print(f"Result for P13: \n {s_13}")


# P14
se_14 = np.array([[0, -1.5708, 0, 2.3562], [1.5708, 0, 0, -2.3562], [0,0,0,1],[0,0,0,0]])
SE3_14 = mr.MatrixExp6(se_14)
print(f"Result for P14: \n {SE3_14}")
                 
# P15
SE3_15 = np.array([[0,-1,0,3],[1,0,0,0],[0,0,1,1,],[0,0,0,1]])
se3_15 = mr.MatrixLog6(SE3_15)
print(f"Result for P15: \n {se3_15}")