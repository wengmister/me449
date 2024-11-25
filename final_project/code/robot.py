from milestone1 import *
from milestone2 import *
from milestone3 import *


class Robot:
    def __init__(self):
        self.box_dim = 0.05
        self.cube_initial = np.array([[1, 0, 0, 1],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, self.box_dim/2],
                                      [0, 0, 0, 1]])
        self.cube_final = np.array([[0, 1, 0, 0],
                                    [-1, 0, 0, -1],
                                    [0, 0, 1, self.box_dim/2],
                                    [0, 0, 0, 1]])
        
        self.Tse_initial = np.array([[0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [-1, 0, 0, 0.5],
                                    [0, 0, 0, 1]])
        
        self.Tce_grasp = np.array([[0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [-1, 0, 0, -0.1],
                                    [0, 0, 0, 1]])
        
        self.Tb0 = np.array([[1, 0, 0, 0.1662],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.0026],
                             [0, 0, 0, 1]])
        
        self.M0e = np.array([[1, 0, 0, 0.033],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0.6546],
                             [0, 0, 0, 1]])
        
        self.Blist = np.array([[0, 0, 1, 0, 0.033, 0],
                               [0, -1, 0, -0.5076, 0, 0],
                               [0, -1, 0, -0.3526, 0, 0],
                               [0, -1, 0, -0.2176, 0, 0],
                               [0, 0, 1, 0, 0, 0]])
        
        # base configs
        l = 0.47/2
        r = 0.0475
        w = 0.3/2
        gamma1 = -np.pi/4
        gamma2 = np.pi/4
        h1 = np.array([1, np.tan(gamma1)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[-w, 1, 0],[l, 0, 1]])/r
        h2 = np.array([1, np.tan(gamma2)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[w, 1, 0],[l, 0, 1]])/r
        h3 = np.array([1, np.tan(gamma1)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[w, 1, 0],[-l, 0, 1]])/r
        h4 = np.array([1, np.tan(gamma2)]) @ np.array([[np.cos(0), np.sin(0)], [-np.sin(0), np.cos(0)]]) @ np.array([[-w, 1, 0],[-l, 0, 1]])/r
        self.H0 = np.vstack([h1, h2, h3, h4])
        self.F3 = np.linalg.pinv(self.H0)
        # Expand self.F3 to a 6x4 matrix by padding with zeros
        expanded_F3 = np.zeros((6, 4))
        expanded_F3[2:5, :self.F3.shape[1]] = self.F3

        # Assign the expanded matrix to F6
        self.F6 = expanded_F3
        
        # Apply a 45 degree rotation about the z-axis to the starting configuration
        # This will hold the actual configuration of the ee.
        self.Tse_actual = np.eye(4)

        # ee trajectory
        self.desired_ee_trajectory = []
        self.actual_trajectory = []

        # youbot states
        self.state_actual = [np.pi/4, -0.1, 0, 0, np.pi/10, -np.pi/2, np.pi/10, 0, 0, 0, 0, 0, 0]
        self.states_planned = []
        self.states_planned.append(self.state_actual)

        self.dt = 0.01
        self.ki = np.eye(6)*0.2
        self.kp = np.eye(6)
        
    def plan_desired_trajectory(self):

        # # move from start configuration to initial configuration
        # packed_1 = mr.ScrewTrajectory(self.Tse_start, self.Tse_initial, 1, 1/0.01, 'quintic')
        # unpacked_1 = unpack(packed_1, 0)
        # for i in unpacked_1:
        #     self.desired_trajectory.append(i)

        planned_traj = TrajectoryGenerator(Tse_initial=self.Tse_initial, 
                                                      Tsc_initial=self.cube_initial, 
                                                      Tsc_final=self.cube_final, 
                                                      Tce_grasp=self.Tce_grasp,
                                                      Tce_standoff=np.array([[0, 0, 1, 0],
                                                                             [0, 1, 0, 0],
                                                                             [-1, 0, 0, 0.3],
                                                                             [0, 0, 0, 1]]), 
                                                      k=1)
        
        for i in planned_traj:
            self.desired_ee_trajectory.append(i)

    def get_current_Tse(self):
        # get Tse_current from actual state: Tsb @ Tb0 @ T0e
        Tsb = mr.RpToTrans(R=mr.MatrixExp3(mr.VecToso3([0, 0, self.state_actual[0]])), p=np.array([self.state_actual[1],self.state_actual[2],0]))
        Tb0 = self.Tb0
        T0e = mr.FKinBody(self.M0e, self.Blist.T, self.state_actual[3:8])
        Tse_current = np.dot(np.dot(Tsb, Tb0), T0e)
        return Tse_current

    def find_jacobian(self):
        # find arm jacobian first
        arm_angles = self.state_actual[3:8]
        # print(f"arm angles: {arm_angles}")
        jac_arm = mr.JacobianBody(self.Blist.T, np.array(arm_angles))

        # find the base jacobian
        # find ee in arm base frame
        T0e = mr.FKinBody(self.M0e, self.Blist.T, arm_angles)

        jac_base = np.dot(mr.Adjoint(np.dot(mr.TransInv(T0e), mr.TransInv(self.Tb0))), self.F6)
        
        #horizontally stack jacobians
        jac = np.hstack([jac_base, jac_arm])

        return jac

    def execute_trajectory(self):
        
        for i in range(len(self.desired_ee_trajectory)-1):
            # print(i)

            # perform feedback control to get 6vector twist
            V_output = FeedbackControl(X= self.get_current_Tse(),
                                        X_d=pack_instance(self.desired_ee_trajectory[i][:-1]), 
                                        X_d_next=pack_instance(self.desired_ee_trajectory[i+1][:-1]), 
                                        K_p=self.kp, 
                                        K_i=self.ki,
                                        dt=self.dt)

            # find jacobian at this configuration and convert to joint values
            jac = self.find_jacobian()

            # find the joint velocities
            q_dot = np.dot(np.linalg.pinv(jac), V_output) # 4x u then 5x thetadot

            # Find the next actual state
            state_output = NextState(self.state_actual, q_dot, self.dt, 1000)
            state_output_with_gripper = np.append(state_output, self.desired_ee_trajectory[i][-1])
            self.states_planned.append(state_output_with_gripper)
            self.state_actual = state_output_with_gripper


if __name__=="__main__":
    robot = Robot()
    robot.plan_desired_trajectory()
    robot.execute_trajectory()
    print(f"current Tse: {robot.get_current_Tse()}")
    print(f"current jacobian: {robot.find_jacobian()}")

    traj_to_csv(robot.desired_ee_trajectory, 'desired.csv')
    # print("Planned states:")
    # print(robot.states_planned)
    traj_to_csv(robot.states_planned, 'output.csv')
    # print(len(robot.desired_trajectory))
    # traj_to_csv(robot.desired_trajectory, 'output.csv')