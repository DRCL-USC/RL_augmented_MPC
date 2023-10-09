import numpy as np
import math
import copy

def bezier_get_joint_poision(t, alpha):
    q = 0
    for i in range(6):
        q += alpha[i] * math.factorial(5) * np.power(t, i) * np.power(1-t, 5-i)/(math.factorial(i) * math.factorial(5-i))
        
    return q

def bezier_get_joint_vel(t,alpha):
    dq = 0
    for i in range(5):
        dq += (alpha[i+1] - alpha[i]) * math.factorial(4) * np.power(t, i) * np.power(1-t, 4-i)/(math.factorial(i) * math.factorial(4-i))

    return dq