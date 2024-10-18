import modern_robotics as mr
import numpy as np
import sympy as sym
np.set_printoptions(precision=2, suppress=True)


J1_b = [0, -1, 0, 2, 0, 3]
J2_b = [0, 0, 0, 1/np.sqrt(5), 0, -2/np.sqrt(5)]
J3_b = [0, -1, 0, 0, 0, 2]


Jacobian_b = np.array([J1_b, J2_b, J3_b]).T
print(Jacobian_b)

f = sym.symbols('f')


# This is the external wrench w.r.t body frame
wrench = [0, -2*f, 0, 0, 0, -f]
wrench_internal = -1 * np.array(wrench)


# When calculating reaction forces, we to use the opposite of the external wrench
torques = np.dot(Jacobian_b.T, wrench_internal)
sym.simplify(torques)
print(torques)