import numpy as np
import modern_robotics as mr
import sympy as sym

# Set NumPy print options to limit float precision to 2 decimal places
np.set_printoptions(precision=2, suppress=True)

def to_comma_separated(array):
    # Convert each row to a comma-separated string
    rows = [', '.join(map(str, row)) for row in array]
    # Join all rows with brackets and commas
    result = '[[' + '], ['.join(rows) + ']]'
    return result

# Function to truncate floats in a list to two decimal places
def truncate_floats(input_list):
    return [round(float(x), 2) for x in input_list]

# Q1
L = 1

M_rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
M_translate = np.array([[L*(2+np.sqrt(3))],[0],[L*(1+np.sqrt(3))]])
M = np.concatenate((M_rot,M_translate),axis=1)
M = np.concatenate((M,np.array([[0,0,0,1]])),axis=0)
print("Result for Q1:")
print(M)

# Q2
j1_s = [0,0,1,0,-L,0]
j2_s = [0,1,0,0,0,L]
j3_s = [0,1,0,L,0,L*(1+np.sqrt(3))]
j4_s = [0,1,0,-L*(np.sqrt(3)-1),0,L*(2+np.sqrt(3))]
j5_s = [0,0,0,0,0,1]
j6_s = [0,0,1,00,-L*(2+np.sqrt(3)),0]

print("Result for Q2:")
# print([truncate_floats(j1_s),truncate_floats(j2_s),truncate_floats(j3_s),truncate_floats(j4_s),truncate_floats(j5_s),truncate_floats(j6_s)])
q2_result = np.array([j1_s,j2_s,j3_s,j4_s,j5_s,j6_s])
# print(q2_result.T)
print(to_comma_separated(q2_result.T))

# Q3 now in b frame
j1_b = [0,0,1,0,L*(1+np.sqrt(3)),0]
j2_b = [0,1,0,L*(np.sqrt(3)+1),0,-L*(1+np.sqrt(3))]
j3_b = [0,1,0,L*(2+np.sqrt(3)),0,-L]
j4_b = [0,1,0,2*L,0,0]
j5_b = [0,0,0,0,0,1]
j6_b = [0,0,1,0,0,0]

print("Result for Q3:")
# print([truncate_floats(j1_b),truncate_floats(j2_b),truncate_floats(j3_b),truncate_floats(j4_b),truncate_floats(j5_b),truncate_floats(j6_b)])
q3_result = np.array([j1_b,j2_b,j3_b,j4_b,j5_b,j6_b])
print(to_comma_separated(q3_result.T))
# Q4
theta_list = [-np.pi/2, np.pi/2, np.pi/3, -np.pi/4, 1, np.pi/6]
T_ee = mr.FKinSpace(M, q2_result.T,theta_list)
T_ee_comma = to_comma_separated(T_ee)
print(f"Result for Q4: \n {T_ee_comma}")

# Q5
T_ee_body = mr.FKinBody(M, q3_result.T,theta_list)
T_ee_body = to_comma_separated(T_ee_body)
print(f"Result for Q5: \n {T_ee_body}")