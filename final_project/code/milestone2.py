import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
import csv

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
    Tse_standoff_init = Tsc_initial @ Tce_standoff
    # Find the grasp configuration in {s}
    Tse_grasp_init = Tsc_initial @ Tce_grasp
    # Final standoff configuration in {s}
    Tse_standoff_final = Tsc_final @ Tce_standoff
    # Final grasp configuration in {s}
    Tse_grasp_final = Tsc_final @ Tce_grasp

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
    packed_1 = mr.ScrewTrajectory(Tse_initial, Tse_standoff_init, t_1, t_1/fps, "quintic")
    unpacked_1 = unpack(packed_1, 0)
    for i in unpacked_1:
        traj.append(i)

    # Move from standoff to grasp configuration
    packed_2 = mr.CartesianTrajectory(Tse_standoff_init, Tse_grasp_init, t_2, t_2/fps, "quintic")
    unpacked_2 = unpack(packed_2, 0)
    for i in unpacked_2:
        traj.append(i)

    # Close the gripper
    packed_3 = mr.CartesianTrajectory(Tse_grasp_init, Tse_grasp_init, t_3, t_3/fps, "quintic")
    unpacked_3 = unpack(packed_3, 1)
    for i in unpacked_3:
        traj.append(i)

    # Move to back to first standoff configuration
    packed_4 = mr.CartesianTrajectory(Tse_grasp_init, Tse_standoff_init, t_4, t_4/fps, "quintic")
    unpacked_4 = unpack(packed_4, 1)
    for i in unpacked_4:
        traj.append(i)

    # Move from grasp to final standoff configuration
    packed_5 = mr.ScrewTrajectory(Tse_standoff_init, Tse_standoff_final, t_5, t_5/fps, "quintic")
    unpacked_5 = unpack(packed_5, 1)
    for i in unpacked_5:
        traj.append(i)

    # Move from final standoff to final grasp configuration
    packed_6 = mr.CartesianTrajectory(Tse_standoff_final, Tse_grasp_final, t_6, t_6/fps, "quintic")
    unpacked_6 = unpack(packed_6, 1)
    for i in unpacked_6:
        traj.append(i)

    # Open the gripper
    packed_7 = mr.CartesianTrajectory(Tse_grasp_final, Tse_grasp_final, t_7, t_7/fps, "quintic")
    unpacked_7 = unpack(packed_7, 0)
    for i in unpacked_7:
        traj.append(i)
    
    # Move back to final standoff configuration
    packed_8 = mr.CartesianTrajectory(Tse_grasp_final, Tse_standoff_final, t_8, t_8/fps, "quintic")
    unpacked_8 = unpack(packed_8, 0)
    for i in unpacked_8:
        traj.append(i)

    return traj

def traj_to_csv(traj, filename='output.csv'):
    """
    Save the generated trajectory data to a CSV file.
    
    Parameters:
    data (list): List of lists containing trajectory
    filename (str): Name of the output CSV file
    """
    
    # Write the data to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the data rows
        writer.writerows(traj)
        
    print(f"Data successfully saved to {filename}")

def main():
    box_dim = 0.025
    Tce_standoff = np.array([[0, 0, 1, 0], 
                             [0, 1, 0, 0], 
                             [-1, 0, 0, 0.3], 
                             [0, 0, 0, 1]])
    Tse_initial = np.array([[0, -1, 0, 0], 
                            [0, 0, 1, 1], 
                            [-1, 0, 0, 0], 
                            [0, 0, 0, 1]])
    Tsc_initial = np.array([[1, 0, 0, 0.5], 
                            [0, 1, 0, 0], 
                            [0, 0, 1, box_dim/2], 
                            [0, 0, 0, 1]])
    Tsc_final = np.array([[0, 1, 0, 0],
                          [-1, 0, 0, -0.5],
                          [0, 0, 1, box_dim/2],
                          [0, 0, 0, 1]])
    Tce_grasp = np.array([[0, 0, 1, 0],
                          [0, 1 ,0 ,0],
                          [-1, 0, 0, 0],
                          [0, 0, 0, 1]])
    test = TrajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, 2)
    # print(test)
    traj_to_csv(test, 'output.csv')

if __name__ == "__main__":
    main()
