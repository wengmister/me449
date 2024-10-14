import modern_robotics as mr
import numpy as np


M = np.array([
    [0, -1, 0, 19],
    [-1, 0, 0, 0],
    [0, 0, -1, -3],
    [0, 0, 0, 1]
])

print(M)


# J1 
J1_s_screw = [0, 0, 1, 0, 0, 0]
J1_b_screw = [0, 0,-1, -19, 0 ,0]

# J2
J2_s_screw = [0,0,1,0,-10,0]
J2_b_screw = [0,0,-1,-9,0,0]

# J3
J3_s_screw = [0,0,1, 0,-19, 0]
J3_b_screw = [0,0,-1,0,0,0]

# J4
J4_s_screw = [0,0,0,0,0,1]
J4_b_screw = [0,0,0,0,0,-1]


# Barrett tech example:

# J1
J1_s = [0,0,1,0,0,0]
J2_s = [0,1,0,0,0,0]
J3_s = [0,0,1,0,0,0]

J4_s = [0,1,0,-550,0,45]
J5_s = [0,0,1,0,0,0]
J6_s = [0,1,0,-850,0,0]
J7_s = [0,0,1,0,0,0]