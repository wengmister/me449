import modern_robotics as mr
import numpy as np
np.set_printoptions(precision=4)

def NextState(current_config: np.array, joint_speeds: np.array, dt: float, max_speed: float) -> np.array:
    """Compute the next state of the robot

    Args:
        current_config (np.array): The current configuration of the robot
        [phi, x, y, j1, j2, j3, j4, j5, w1, w2, w3, w4]
        joint_speeds (np.array): The speed of each joint
        [w1, w2, w3, w4, j1, j2, j3, j4, j5]
        dt (float): The time step
        max_speeds (np.array): The maximum speed of each joint

    Returns:
        np.array: The next state of the robot - 12 vector
    """
    # Extract the arm and wheel configuration
    arm_config = current_config[3:8]
    # print(f"arm_config={arm_config}")
    wheel_config = current_config[-4:]
    # print(f"wheel_config={wheel_config}")

    # Extract speeds
    arm_speeds = joint_speeds[-5:]
    # print(f"arm_speeds={arm_speeds}")
    wheel_speeds = joint_speeds[:4]
    # print(f"wheel_speeds={wheel_speeds}")

    # Check if speeds go beyond the maximum speed
    for i in range(len(arm_speeds)):
        if arm_speeds[i] > max_speed:
            arm_speeds[i] = max_speed
        if arm_speeds[i] < -max_speed:
            arm_speeds[i] = -max_speed
    
    for i in range(len(wheel_speeds)):
        if wheel_speeds[i] > max_speed:
            wheel_speeds[i] = max_speed
        if wheel_speeds[i] < -max_speed:
            wheel_speeds[i] = -max_speed

    # Compute odometry
    odometry_update = compute_odometry(wheel_speeds)

    # Compute the next state
    next_arm_config = arm_config + arm_speeds * dt
    next_wheel_config = wheel_config + wheel_speeds * dt
    next_chassis_config = current_config[:3] + odometry_update * dt
    
    next_state = np.concatenate([next_chassis_config, next_arm_config, next_wheel_config])

    return next_state

def compute_odometry(wheel_speeds: np.array) -> np.array:
    """Compute the odometry of the robot

    Args:
        wheel_speeds (np.array): The speed of each wheel

    Returns:
        np.array: The odometry update for the robot
    """
    l = 0.47/2
    r = 0.0475
    w = 0.3/2
    gamma1 = -np.pi/4
    gamma2 = np.pi/4
    h1 = np.array([1, np.tan(gamma1)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[-w, 1, 0],[l, 0, 1]])/r
    h2 = np.array([1, np.tan(gamma2)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[w, 1, 0],[l, 0, 1]])/r
    h3 = np.array([1, np.tan(gamma1)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[w, 1, 0],[-l, 0, 1]])/r
    h4 = np.array([1, np.tan(gamma2)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[-w, 1, 0],[-l, 0, 1]])/r
    H0 = np.vstack([h1, h2, h3, h4])
    theta_del = wheel_speeds.T
    V_b = np.linalg.pinv(H0) @ theta_del


    return np.array([V_b[0], V_b[1], V_b[2]]) # V_b = [w_z, v_x, v_y]

if __name__ == "__main__":
    # Test the NextState function
    test_speed = np.array([-10,10,-10,10])
    print(f"testing odom with input {test_speed}: {compute_odometry(test_speed)}")
    
    test_config = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    test_joint_speed = np.array([6,3,5,4,1,5,3,5,2])
    test_values = []

    for i in range(100):
        next_config = NextState(test_config, test_joint_speed, 0.1, 10)
        print(f"testing config with: {test_config}: {next_config}")
        test_config = next_config
        test_values.append(next_config)

    #plot all joint values
    import matplotlib.pyplot as plt
    test_values = np.array(test_values)
    plt.plot(test_values[:,:])
    plt.show()