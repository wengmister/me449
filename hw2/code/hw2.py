import modern_robotics as mr
import numpy as np
import math

# The IKinBodyIterates function:
def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot
       iteratively using newton-raphson method

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist,
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
        or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    
    traj = np.array([thetalist])
    ee_pos = np.array(mr.FKinBody(M, Blist, thetalist))[:3,3].T
    err_angle = np.linalg.norm([Vb[0], Vb[1], Vb[2]])
    err_pos = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
    err_list = np.array([err_angle, err_pos])
    

    while err and i < maxiterations:
        thetalist = thetalist \
            + np.dot(np.linalg.pinv(mr.JacobianBody(Blist,
                                                    thetalist)), Vb)
        
        thetalist = [math.atan2(np.sin(theta), np.cos(theta)) for theta in thetalist]
        thetalist = np.array(thetalist)
        
        i = i + 1
        Vb \
            = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist,
                                                                       thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
            or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
        
        print(f"Iteration {i}:\n")
        print("joint vector:")
        print(f"{thetalist}\n")
        print("SE(3) end-effector config:\n")
        print(f"{mr.FKinBody(M, Blist, thetalist)}\n")

        print(f"          error twist V_b: {Vb}")
        print(f"angular error ||omega_b||: {np.linalg.norm([Vb[0], Vb[1], Vb[2]])}")
        print(f"     linear error ||v_b||: {np.linalg.norm([Vb[3], Vb[4], Vb[5]])}\n")

        traj = np.vstack([traj, thetalist])
        ee_pos = np.vstack([ee_pos, np.array(mr.FKinBody(M, Blist, thetalist))[:3,3].T])
        err_list = np.vstack([err_list, np.array([np.linalg.norm([Vb[0], Vb[1], Vb[2]]), np.linalg.norm([Vb[3], Vb[4], Vb[5]])])])
        print(f"Trajectory:\n{traj}\n")

    # Save to csv file
    np.savetxt("IKinBodyIterates.csv", traj, delimiter=",")

    return (thetalist, not err, i, traj, ee_pos, err_list)


if __name__=="__main__":
    # Example:
    J1_B = np.array([0,0,1,0,3,0])
    J2_B = np.array([0,0,1,0,2,0])
    J3_B = np.array([0,0,1,0,1,0])

    Blist = np.column_stack([J1_B, J2_B, J3_B])

    T_sb = np.array([[1, 0, 0, 3],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    T_sd = np.array([[-0.585, -0.811, 0, 0.076],[0.811, -0.585, 0, 2.608],[0,0,1,0],[0,0,0,1]])
    result = IKinBodyIterates(Blist, T_sb, T_sd, np.array([np.pi/4, np.pi/4, np.pi/4]), 0.01, 0.001)