import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt

"""
We need to achieve the following 8 trajectories:


1. A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
2. A trajectory to move the gripper down to the grasp position.
3. Closing of the gripper.
4. A trajectory to move the gripper back up to the "standoff" configuration.
5. A trajectory to move the gripper to a "standoff" configuration above the final configuration.
6. A trajectory to move the gripper to the final configuration of the object.
7.Opening of the gripper.
8. A trajectory to move the gripper back to the "standoff" configuration.
"""

def unpack(traj: list[np.array], gripper_state:int) -> list[np.array]:
    """unpack the trajectory SE(3) into list with gripper state

    Args:
        traj (list[SE(3)]): a list of SE(3) matrices trajectory
        gripper_state (int): 0 for open, 1 for closed
    """
    unpacked = []
    for T in traj:
        T_i = T[:3, :3].flatten().tolist() + T[:3, 3].tolist()
        unpacked.append(T_i + [gripper_state])

    return unpacked



def TrajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, k):
    """generate the reference trajectory for the end-effector frame {e}.

    Args:
        Tse_initial (SE(3)): The initial configuration of the end-effector in the reference trajectory
        Tsc_initial (SE(3)): The cube's initial configuration
        Tsc_final (SE(3)): The cube's desired final configuration
        Tce_grasp (SE(3)): The end-effector's configuration relative to the cube when it is grasping the cube
        Tce_standoff (SE(3)): The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube
        k (int): The number of trajectory reference configurations per 0.01 seconds
    """
    traj = []

    # Find the first standoff configuration in {s}
    Tse_standoff = Tsc_initial @ Tce_standoff
    # Find the grasp configuration in {s}
    Tse_grasp = Tsc_initial @ Tce_grasp

    fps = 0.01/k
    # Total frames = time / (0.01/k)
    t_1 = 2
    t_2 = 1
    t_3 = 0.7
    t_4 = 1
    t_5 = 2
    t_6 = 1
    t_7 = 0.7
    t_8 = 1

    # Move from initial configuration to standoff configuration
    packed_1 = mr.CartesianTrajectory(Tse_initial, Tse_standoff, t_1, t_1/fps, "quintic")
    unpacked_1 = unpack(packed_1, 0)
    traj.append(unpacked_1)

    return traj

def main():
    Tce_standoff = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    test = TrajectoryGenerator(np.eye(4), np.eye(4), np.eye(4), np.eye(4), Tce_standoff, 2)
    print(test)

if __name__ == "__main__":
    main()
