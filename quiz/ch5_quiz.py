import modern_robotics as mr
import numpy as np

# Q1
s1_s = [0,0,1,0,0,0]
s2_s = [0,0,1,0,-1,0]
s3_s = [0,0,1,np.sqrt(2),-1-np.sqrt(2),0]
J_s = np.array([s1_s.T,s2_s.T,s3_s.T])
print("Result for Q1:")
print(J_s)