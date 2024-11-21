import numpy as np
import matplotlib.pyplot as plt


target = 3 #rad
current = 0 #rad

dt = 0.1

kp = 0.2
ki = 0.1

traj = []
for i in range(0,100,1):
    error = target - current
    # P control
    P = kp * error
    current = current + P

    print(f"Current: {current}")
    traj.append(current)

plt.plot(traj)
plt.show()