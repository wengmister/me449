import modern_robotics as mr
import numpy as np

def FeedbackControl(X, X_d, X_d_next, K_p, K_i, dt):
    """
    Compute the feedback control for the robot.

    Args:
        X (SE(3)): The current end-effector configuration
        X_d (SE(3)): The desired end-effector configuration
        X_d_next (SE(3)): The desired end-effector configuration at the next time step
        K_p (np.array): The proportional gain - should be 6x6
        K_i (np.array): The integral gain - should be 6x6
        dt (float): The time step

    Returns:
        np.array: The control output
    """
    # Compute the error twist
    X_err = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(X), X_d)))
    # print(f"X_err={X_err}")

    # Compute feedforward twist
    V_d = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(X_d), X_d_next))) / dt
    # print(f"V_d={V_d}")

    # Compute the feedforward 
    ff_adj = mr.Adjoint(np.dot(mr.TransInv(X), X_d))

    # Compute the control output
    V = np.dot(ff_adj, V_d) + np.dot(K_p, X_err) + np.dot(K_i, X_err * dt)
    
    return V

if __name__ == "__main__":
    # Define the current configuration
    X = np.array([[0.17, 0, 0.985, 0.387],
                  [0, 1, 0, 0],
                  [-0.985, 0, 0.170, 0.570],
                  [0, 0, 0, 1]])

    # Define the desired configuration
    X_d = np.array([[0, 0, 1, 0.5],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0.5],
                    [0, 0, 0, 1]])

    # Define the desired configuration at the next time step
    X_d_next = np.array([[0, 0, 1, 0.6],
                         [0, 1, 0, 0],
                         [-1, 0, 0, 0.3],
                         [0, 0, 0, 1]])

    # Define the gains
    K_p = np.zeros(6)
    K_i = np.zeros(6)

    # Define the time step
    dt = 0.01

    # Compute the control output
    V = FeedbackControl(X, X_d, X_d_next, K_p, K_i, dt)
    print(V)
    # Expected output: [ 0.  0.  0.  0.  0.  0.]
    # The control output is zero because the current configuration is equal to the desired configuration.