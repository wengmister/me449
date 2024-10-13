import modern_robotics as mr
import numpy as np
np.set_printoptions(precision=2, suppress=True)

# finding T_ab
T_ab = np.array([[0,0,1,0],[1,0,0,4],[0,1,0,-2],[0,0,0,1]])

# Screw in a
screw_a = mr.ScrewToAxis([0,0,0],[0,0,1],5)
print(f"Screw_a has: \n {screw_a}")

# Screw in b
T_ba = mr.TransInv(T_ab)
T_ba_adj = mr.Adjoint(T_ba)

screw_b_by_adj = np.dot(T_ba_adj, screw_a)
print(f"Screw_b by adjoint matrix operation: \n {screw_b_by_adj}")

screw_b = mr.ScrewToAxis([-4,2,0],[0,1,0],5)
print(f"Screw_b by inspection: \n {screw_b}")

# Find spatial velocity:
screw_a_se3 = mr.VecTose3(screw_a)
theta = np.pi

displacement_se3 = screw_a_se3 * theta
displacement_SE3 = mr.MatrixExp6(displacement_se3)

final_T_ab = np.dot(displacement_SE3, T_ab)
print(f"in A frame: \n {final_T_ab}")


screw_b_se3 = mr.VecTose3(screw_b)
displacement_se3_b = screw_b_se3 * theta
displacement_SE3_b = mr.MatrixExp6(displacement_se3_b)

final_T_ab_b = np.dot(T_ab, displacement_SE3_b)
print(f"In B frame: \n {final_T_ab_b}")



######################### Practice 2 ##################################

# exp([V_a] * t) * T_ab = T_ac
# T_cb = T_ac ^ T * T_ab
# log(T_ac x T_ab^T)/t

######################### Practice 3 ##################################

S_b = mr.ScrewToAxis([0,0,0],[0,0,1],1)
print(S_b)