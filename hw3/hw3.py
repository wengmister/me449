import numpy as np
import modern_robotics as mr
from tqdm import tqdm
import ur5_parameters
ur5 = ur5_parameters.UR5()


# Puppet function:
def puppet_q1(thetalist, dthetalist, g, Mlist, Slist, Glist, t, dt, damping, stiffness, restLength):
    """
    Simulate a robot under damping and spring reaction. Q1: free falling in gravity

    Args:
        thetalist (np.array): n-vector of initial joint angles (rad)
        dthetalist (np.array): n-vector of initial joint velocities (rad/s)
        g (np.array): 3-vector of gravity in s frame (m/s^2)
        Mlist (np.array): 8 frames of link configuration at home pose
        Slist (np.array): 6-vector of screw axes at home configuration
        Glist (np.array): Spatial inertia matrices of the links
        t (float): total simulation time (s)
        dt (float): simulation time step (s)
        damping (float): viscous damping coefficient (Nmn/rad)
        stiffness (float): spring stiffness coefficient (N/m)
        restLength (float): length of the spring at rest (m)
    Returns:
        thetamat (np.array): N x n matrix of joint angles (rad). Each row is a set of joint angles
        dthetamat (np.array): N x n matrix of joint velocities (rad/s). Each row is a set of joint velocities
    """
    # Initialize 
    N = int(t/dt)
    n = len(thetalist)
    thetamat = np.zeros((N + 1, n))
    dthetamat = np.zeros((N + 1, n))
    thetamat[0] = thetalist
    dthetamat[0] = dthetalist

    for i in tqdm(range(N)):
        i_acc = mr.ForwardDynamics(thetalist, dthetalist, np.zeros(n), g, np.zeros(n), Mlist, Glist, Slist) 
        i_pos, i_vel = mr.EulerStep(thetalist, dthetalist, i_acc, dt)
        thetamat[i + 1] = i_pos
        dthetamat[i + 1] = i_vel

        # Update
        thetalist = i_pos
        dthetalist = i_vel
    
    return thetamat, dthetamat

def referencePos(t):
    """
    Generate a reference position for springPos 

    Args:
        t (float): current time (s)
    Returns:
        np.array: 3-vector of reference position
    """
    return np.array([0, 0, 0])


if __name__ == "__main__":
    print(np.zeros(6))

    q1_thetalist0 = np.array([0, 0, 0, 0, 0, 0])
    q1_dthetalist0 = np.array([0, 0, 0, 0, 0, 0])
    g = np.array([0, 0, -9.81])

    print(f"test{np.array(ur5.Slist)[:, 1]}")

    print("thetalist shape:", q1_thetalist0.shape)
    print("dthetalist shape:", q1_dthetalist0.shape)
    print("g shape:", g.shape)
    print("Mlist individual shapes:", [np.array(M).shape for M in ur5.Mlist])
    print("Slist shape:", len(ur5.Slist))
    print("Glist individual shapes:", [G.shape for G in ur5.Glist])

    q1_thetamat, _ = puppet_q1(q1_thetalist0, q1_dthetalist0, g, ur5.Mlist, ur5.Slist, ur5.Glist, 5, 0.01, 0, 0, 0)

