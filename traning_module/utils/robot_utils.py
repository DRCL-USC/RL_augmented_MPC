""" Robot utils """
import numpy as np
import pybullet

 ######################################################################################
 # Formatting FR positions to all legs
 ######################################################################################

def FormatFRtoAll(FR_pos):
	"""For a quadruped, format position in FR frame to all other limbs.
	Takes in FR_pos as numpy array, returns list of numpy arrays for the same FR_pos in each frame. """
	FL_pos = np.array( [ FR_pos[0], -FR_pos[1], FR_pos[2]])
	RR_pos = np.array( [-FR_pos[0],  FR_pos[1], FR_pos[2]])
	RL_pos = np.array( [-FR_pos[0], -FR_pos[1], FR_pos[2]])
	return [ FR_pos, FL_pos, RR_pos, RL_pos ]


def format_FR_ranges_to_all(FR_low,FR_upp):
	"""Takes in low/upp ranges for FR and returns corresponding low/upp
	ranges for each of the other legs. Note differences in min/max.

	Examples: 
	consider LEG frame (relative to hip):
	FR_low_leg = [-0.1, -0.13, -0.45]
	FR_upp_leg = [ 0.1, -0.03, -0.35]

	and then body frame (relative to body COM):
	FR_low = [0.1, -0.2 , -0.5]
	FR_upp = [0.5, -0.03, -0.1]

	Also, consider the following would get the min/max wrong:
	# IK_POS_LOW = np.concatenate(FormatFRtoAll(FR_low))
	# IK_POS_UPP = np.concatenate(FormatFRtoAll(FR_upp))
	"""
	FL_low = [ FR_low[0], -FR_upp[1], FR_low[2]]
	FL_upp = [ FR_upp[0], -FR_low[1], FR_upp[2]]

	RR_low = [-FR_upp[0],  FR_low[1], FR_low[2]]
	RR_upp = [-FR_low[0],  FR_upp[1], FR_upp[2]]

	RL_low = [-FR_upp[0], -FR_upp[1], FR_low[2]]
	RL_upp = [-FR_low[0], -FR_low[1], FR_upp[2]]

	POS_LOW = np.concatenate(([FR_low,FL_low,RR_low,RL_low]))
	POS_UPP = np.concatenate(([FR_upp,FL_upp,RR_upp,RL_upp]))

	return POS_LOW, POS_UPP

######################################################################################
# Math utils
######################################################################################
def QuaternionToOrientationMatrix(quat_orn):
	""" Given quaternion, return orientation matrix """
	return np.asarray(pybullet.getMatrixFromQuaternion(quat_orn)).reshape((3,3))


######################################################################################
# General robot functionalities
######################################################################################