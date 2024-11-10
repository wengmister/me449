import numpy as np
import modern_robotics as mr
from tqdm import tqdm
import ur5_parameters
import matplotlib.pyplot as plt
ur5 = ur5_parameters.UR5()


# Puppet functions:
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
    hmat = np.zeros(N + 1)
    tmat = np.zeros(N + 1)
    vmat = np.zeros(N + 1)
    thetamat[0] = thetalist
    dthetamat[0] = dthetalist

    for i in tqdm(range(N)):
        i_acc = mr.ForwardDynamics(thetalist, dthetalist, np.zeros(n), g, np.zeros(n), Mlist, Glist, Slist) 
        i_pos, i_vel = mr.EulerStep(thetalist, dthetalist, i_acc, dt)
        thetamat[i + 1] = i_pos
        dthetamat[i + 1] = i_vel

        H_i = compute_hamiltonian(i_pos, i_vel, g, Mlist, Glist, Slist)
        hmat[i] = H_i[0]
        tmat[i] = H_i[1]
        vmat[i] = H_i[2]

        # Update
        thetalist = i_pos
        dthetalist = i_vel
    
    return thetamat, dthetamat, hmat, tmat, vmat

def puppet_q2(thetalist, dthetalist, g, Mlist, Slist, Glist, t, dt, damping, stiffness, restLength):
    """
    Simulate a robot under damping and spring reaction. Q2: Adding damping to robot. Damping causes a tau = - dtheta * damping

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
        tau_damping = - damping * dthetalist
        i_acc = mr.ForwardDynamics(thetalist, dthetalist, tau_damping, g, np.zeros(n), Mlist, Glist, Slist) 
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

def compute_hamiltonian(thetalist, dthetalist, g, Mlist, Glist, Slist):
    """
    Compute the Hamiltonian (total energy) of the UR5 robot.
    
    Args:
        thetalist: n-vector of joint variables
        dthetalist: n-vector of joint velocities
        g: 3-vector of gravitational acceleration
        Mlist: List of link frames i relative to i-1 at home position
        Glist: Spatial inertia matrices Gi of the links
        Slist: 6xn matrix of screw axes in space frame
    
    Returns:
        H: Hamiltonian (scalar) representing total energy
        T: Kinetic energy (scalar)
        V: Potential energy (scalar)
    """
    # Compute Kinetic Energy: T = (1/2) * dq^T * M(q) * dq
    # Get mass matrix
    M = mr.MassMatrix(thetalist, Mlist, Glist, Slist)
    # Compute kinetic energy
    T = 0.5 * np.dot(dthetalist.T, np.dot(M, dthetalist))
    
    Slist = np.array(Slist)

    # Compute Potential Energy: V = -sum(mi*g^T*pi)
    V = 0
    # Get end effector transformation matrix for current configuration
    for i in range(len(thetalist)):
        # Get current transformation matrix up to link i
        # print(f"current Mlist: {Mlist[:i+1]}")
        # print(f"current Slist: {Slist[:,:i+1]}")
        # print(f"current thetalist: {thetalist[:i+1]}")
        Mi = np.eye(4)
        for j in range(i):
            Mi = Mi @ Mlist[j]
        Ti = mr.FKinSpace(Mi, Slist[:,:i+1], thetalist[:i+1])
        # print(f"Ti = {Ti}")
        Ti = np.array(Ti).reshape(4,4)
        # Extract position vector (first 3 elements of last column)
        pi = np.array(Ti[:3,3])
        # print(pi)
        # Get mass of link i (from spatial inertia matrix)
        mi = Glist[i][3,3]  # Mass is stored in (3,3) position of spatial inertia matrix
        # Add contribution to potential energy
        V += mi * np.dot(-g, pi.T)
    
    # Compute Hamiltonian: H = T + V
    H = T + V
    
    return H, T, V


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

    q1_thetamat, _ , q1_hmat, q1_tmat, q1_vmat = puppet_q1(q1_thetalist0, q1_dthetalist0, g, ur5.Mlist, ur5.Slist, ur5.Glist, 5, 0.01, 0, 0, 0)
    print(q1_thetamat.shape)
    plt.figure()
    plt.plot(q1_hmat, label = "Hamiltonian")
    plt.plot(q1_tmat, label = "Kinetic Energy")
    plt.plot(q1_vmat, label = "Potential Energy")
    plt.legend()
    plt.show()
