import os
import numpy as np
from pybullet_utils import transformations
import pybullet as p

""" Class for loading and accessing optimal trajectory. """

file_root =  os.path.join(os.path.dirname(__file__), '')
defaultFilename = 'data/jumpingFull2.csv'

# these are the indices at which 2D information can be extracted from the full_state
# base pos is [0:3]
# base RPY is [3:6]
# base LinVel is [6:9]
# base AngVel is [9:12]
# jointPos is [12:24]
#	- within jointPos, 
#		-FR is [0,1,2] (qFront is [1,2])
#		-RR is [6,7,8] (qRear  is [7,8])
# jointVel is [24:36]
jointPosOffset = 12
jointVelOffset = 24
indices2D = np.array([0,2,
					  4,
					  6,8,
					  10,
					  jointPosOffset+1,jointPosOffset+2,jointPosOffset+7,jointPosOffset+8,
					  jointVelOffset+1,jointVelOffset+2,jointVelOffset+7,jointVelOffset+8])



class TrajTaskMotionData(object):
	""" Class to represent optimal trajectory, including functions to query particular states, etc. 
	Notes:
	-may want to read in a different format, and should also work for arbitrary dt. 
	i.e. google uses JSON format with all information in that file.
	"""

	# indices for full state
	BASE_POS_INDICES = 0 + np.arange(3)
	BASE_ORN_INDICES = 3 + np.arange(3)
	BASE_LINVEL_INDICES = 6 + np.arange(3)
	BASE_ANGVEL_INDICES = 9 + np.arange(3)
	JOINT_POS_INDICES = 12 + np.arange(12)
	JOINT_VEL_INDICES = 24 + np.arange(12)
	# indices for Cartesian state data
	FOOT_POS_INDICES = 36 + np.arange(12)
	FOOT_VEL_INDICES = 48 + np.arange(12)
	FOOT_FORCE_INDICES = 60 + np.arange(12)

	# with quaternion
	# BASE_ORN_INDICES = 3 + np.arange(4)
	# BASE_LINVEL_INDICES = 7 + np.arange(3)
	# BASE_ANGVEL_INDICES = 10 + np.arange(3)
	# JOINT_POS_INDICES = 13 + np.arange(12)
	# JOINT_VEL_INDICES = 25 + np.arange(12)
	# # indices for Cartesian state data
	# FOOT_POS_INDICES = 37 + np.arange(12)
	# FOOT_VEL_INDICES = 49 + np.arange(12)
	# FOOT_FORCE_INDICES = 61 + np.arange(12)

	# 2D env indices for full state
	BASE_POS_INDICES_2D = np.array([0,2])
	BASE_ORN_INDICES_2D = np.array([4]) 
	BASE_LINVEL_INDICES_2D = np.array([6,8])
	BASE_ANGVEL_INDICES_2D = np.array([10])
	JOINT_POS_INDICES_2D = 12 + np.array([1,2,7,8])
	JOINT_VEL_INDICES_2D = 24 + np.array([1,2,7,8])


	def __init__(self,filename, dt=0.001, useCartesianData=False, useQuaternion=False):
		"""
		Take in filename, timestep (or should read from JSON) 
		"""
		self.useCartesianData = useCartesianData
		self.useQuaternion = useQuaternion
		# Make different functions if the format changes. May want to read from JSON as well/
		# self.loadTrajDataFromCSV(file_root + filename)
		# if useCartesianData:
		# 	self.loadCartesianTrajDataFromCSV(file_root + filename)
		self.loadTrajDataFromCSV(filename)
		if useCartesianData:
			self.loadCartesianTrajDataFromCSV(filename)
		self._setFullStateFromCSV()
		self.dt = dt

		self._filename = filename


	def loadTrajDataFromJSON(self,filename):
		"""TODO: change matlab to output data in JSON format so 
		we do not have to manually specify time step, just read in.
		"""
		raise NotImplementedError

	def _setFullStateFromCSV(self):
		""" Call after loadTrajDataFromCSV(),
		want full state in terms of baseXYZ, baseRPY, baseLinvel, baseAngVel, full_joint states pos(12) vel(12)
		"""
		self.baseXYZ = np.zeros((3,self.trajLen))
		if self.useQuaternion:
			self.baseOrn = np.zeros((4,self.trajLen))
		else:
			self.baseRPY = np.zeros((3,self.trajLen))
		self.baseLinVel = np.zeros((3,self.trajLen))
		self.baseAngVel = np.zeros((3,self.trajLen))
		self.jointsPos = self.q.copy() 
		self.jointsVel = self.dq.copy()

		self.baseXYZ[0,:] = self.xzphi[0,:].copy()
		self.baseXYZ[2,:] = self.xzphi[1,:].copy()
		if self.useQuaternion:
			for i in range(self.trajLen):
				self.baseOrn[i,:] = p.getQuaternionFromEuler([0, self.baseRPY[1,i], 0])
		else:
			self.baseRPY[1,:] = self.xzphi[2,:].copy()

		self.baseLinVel[0,:] = self.dxzphi[0,:].copy()
		self.baseLinVel[2,:] = self.dxzphi[1,:].copy()
		self.baseAngVel[1,:] = self.dxzphi[2,:].copy()

		self.full_state = np.vstack((self.baseXYZ, # 3
									 self.baseRPY, # 3
									 self.baseLinVel, # 3
									 self.baseAngVel, # 3
									 self.jointsPos, #12
									 self.jointsVel, #12
									 #self.taus 
									 ))

		if self.useCartesianData:
			self.full_state = np.vstack((self.baseXYZ, 
									 self.baseRPY, 
									 self.baseLinVel,
									 self.baseAngVel,
									 self.jointsPos,
									 self.jointsVel,
									 self.foot_pos, #12
									 self.foot_vel, #12
									 self.foot_force, #12
									 #self.taus 
									 ))

	def writeoutTraj(self, path):
		"""Write out full_state """
		head, tail = os.path.split(self._filename)
		new_basename = tail[:-4] + '_full_state.csv'
		output_stack = np.vstack(( self.full_state, self.taus ))
		#np.savetxt(os.path.join(path,new_basename),self.full_state.T,delimiter=',')
		np.savetxt(os.path.join(path,new_basename),output_stack.T,delimiter=',')

		# tau_basename = tail[:-4] + '_tau.csv'
		# np.savetxt(os.path.join(path,tau_basename),self.taus.T,delimiter=',')

	def loadTrajDataFromCSV(self,filename):
		""" Load in trajectory from matlab output (CSV or txt file). 

		format is [Qs (q; dq); taus] from optimization, where:
		q = [x, y, phi, front_thigh, front_calf, back_thigh, back_calf]
		taus = [tau_front_thigh, tau_front_calf, tau_back_thigh, tau_back_calf]

		Notes:
		-angles must be mapped, currently outside of ranges (change to negative)
		-torques should be divided by 2 since the optimization is 2D
		-add z offset (0.2 m)

		"""
		trajData = np.loadtxt(filename,delimiter=',')
		self.trajLen = trajLen = trajData.shape[1]

		#optimization data
		self.xzphi = trajData[0:3,:]
		# print("self.xzphi length:", self.xzphi.shape[0] )
		# print("self.xzphi:", self.xzphi[:,1])
		# add z offset (h_com_init in matlab opt) # 0.1526 # 0.163 with feet
		# self.xzphi[1] = self.xzphi[1] + 0.173  #0.173 # for aliengo+ 0.22
		self.xzphi[1] = self.xzphi[1] + 0.163  #0.173 # for aliengo+ 0.22
		# self.xzphi[1] = self.xzphi[1] + 0.1526  #0.173 # for aliengo+ 0.22
		self.dxzphi = trajData[7:10,:]
		q = trajData[3:7,:]
		dq= trajData[10:14,:]
		torques = trajData[14:,:]

		# map from 2D optimization to 3D aliengo in pybullet
		qFront = -q[0:2,:]
		qRear  = -q[2:4,:]
		dqFront = -dq[0:2,:]
		dqRear  = -dq[2:4,:]

		torquesFront = -torques[0:2,:]
		torquesRear  = -torques[2:4,:]

		# fill in full q, dq, torque arrays
		frontIndices = np.array([1,2,4,5])
		rearIndices  = np.array([7,8,10,11])
		self.q = np.zeros((12,trajLen))
		self.dq = np.zeros((12,trajLen))
		self.taus = np.zeros((12,trajLen))

		self.q[frontIndices,:] = np.vstack((qFront,qFront))
		self.q[rearIndices,:]  = np.vstack((qRear,qRear))
		self.dq[frontIndices,:] = np.vstack((dqFront,dqFront))
		self.dq[rearIndices,:]  = np.vstack((dqRear,dqRear))
		self.taus[frontIndices,:] = np.vstack((torquesFront,torquesFront))
		self.taus[rearIndices,:]  = np.vstack((torquesRear,torquesRear))

		# since optimization was 2D 
		self.taus = self.taus / 2.0

		#self.full_state = np.vstack((self.xzphi,self.dxzphi, self.q,self.dq))
		self.full_state2D = np.vstack((self.xzphi,self.dxzphi, qFront, qRear, dqFront, dqRear))
		#self.taus2D = 

	def loadCartesianTrajDataFromCSV(self,filename):
		"""Load in Cartesian space data, if available, in leg frame

		format is [foot_pos_leg (Front xz, Rear xz); 
				   foot_vel_leg (Front dx,dz, Rear dx,dz);
				   Ff (Ffx, Ffz); 
				   Fr (Frx, Frz)]
		"""
		cartesian_filename = filename[:-4] + '_cartesian.csv'
		try:
			trajData = np.loadtxt(cartesian_filename,delimiter=',')
		except:
			raise ValueError('Unable to load cartesian space data for this trajectory.')

		# frontIndices = np.array([1,2,4,5])
		# rearIndices  = np.array([7,8,10,11])
		# xz positions
		frontIndices = np.array([0,2,3,5])
		rearIndices  = np.array([6,8,9,11])

		self.foot_pos = np.zeros((12,self.trajLen))
		self.foot_vel = np.zeros((12,self.trajLen))
		self.foot_force = np.zeros((12,self.trajLen))

		self.foot_pos[frontIndices,:] = np.vstack((trajData[0:2,:],trajData[0:2,:]))
		self.foot_pos[rearIndices,:]  = np.vstack((trajData[2:4,:],trajData[2:4,:]))
		self.foot_vel[frontIndices,:] = np.vstack((trajData[4:6,:],trajData[4:6,:]))
		self.foot_vel[rearIndices,:]  = np.vstack((trajData[6:8,:],trajData[6:8,:]))
		self.foot_force[frontIndices,:] = np.vstack((trajData[8:10,:], trajData[8:10,:]))
		self.foot_force[rearIndices,:]  = np.vstack((trajData[10:12,:],trajData[10:12,:]))

		# offset y positions
		y_indices = np.array([1,4,7,10])
		hip_length = 0.0838
		self.foot_pos[y_indices,:] = hip_length * np.array([[-1, 1, -1, 1]]).T * np.ones((4,self.trajLen))

		# since optimization was 2D 
		self.foot_force = self.foot_force / 2

	##############################################################################
	## similar to google motion_imitation to get trajectory states
	##############################################################################
	def get_duration(self):
		""" Get the duration (in seconds) of the entire trajectory. 
		trajLen -1 since start at 0 """
		return (self.trajLen-1) * self.dt

	def get_time_at_index(self, index):
		"""  Get the time of a particular trajectory index. """
		return index * self.dt

	def calc_traj_phase(self,t):
		""" Given a time t, return what percent of the trajectory this represents.
		This percent is a scalar in [0,1].
		"""
		# if t > self.get_duration() or t<0:
		# 	raise ValueError("Time outside of trajectory range.") 
		return np.clip( t/self.get_duration(), 0, 1)

	def calc_blend_idx(self, t):
		""" Given a time t, find where it falls inside the trajectory (return relevant traj indices).
		Return:
		(1) f0: start index for blending
		(2) f1: end index for blending
		(3) blend: interpolation value to blend between the two indices (i.e. which are we closer to?)
		"""
		dur = self.get_duration()
		phase = self.calc_traj_phase(t)

		if t <= 0:
			f0 = 0
			f1 = 0
			blend = 0
		elif t >= dur:
			f0 = self.trajLen - 1
			f1 = self.trajLen - 1
			blend = 0
		else:

			f0 = int(phase * (self.trajLen - 1))
			f1 = min(f0 + 1, self.trajLen  - 1)

			norm_time = phase * dur
			time0 = self.get_time_at_index(f0)
			time1 = self.get_time_at_index(f1)
			#print('norm_time, time0, time1',norm_time, time0, time1)
			assert (norm_time >= time0 - 1e-5) and (norm_time <= time1 + 1e-5)

			blend = (norm_time - time0) / (time1 - time0)

		return f0, f1, blend

	def blend_states(self, stateIndex0, stateIndex1, blend):
		""" Linearly interpolate between two consecutive states from the trajectory. Use blend %.
		Returns: blended state
		Notes: 
		Careful with interpolating rotation angles due to discontinuities from 0 to 2*pi
		"""
		#root_rot0 = self.baseRPY[:, stateIndex0]
		#root_rot1 = self.baseRPY[:, stateIndex1]

		# convert angles to quaternion, interpolate in quaternion space, then move back to euler
		# -this should avoid poor interpolation between jumps from 0 to 2*pi
		#root_rot0_quaternion = transformations.quaternion_from_euler(root_rot0[0],root_rot0[1],root_rot0[2])
		#root_rot1_quaternion = transformations.quaternion_from_euler(root_rot1[0],root_rot1[1],root_rot1[2])
		# blend_root_rot_quaternion = transformations.quaternion_slerp(root_rot0_quaternion, 
		# 															 root_rot1_quaternion, 
		# 															 blend)
		# transform back to euler
		""" TODO: verify this is reasonable, possible source of bugs.

		EDIT: using this quaternion blend is currently a BUG. ignore for now.
		"""
		#blend_root_rot = transformations.euler_from_quaternion(blend_root_rot_quaternion)
		#print('verify this is reasonable: orig0',root_rot0, 'orig1', root_rot1, 'blended', blend_root_rot)

		blended_state = (1.0 - blend) * self.full_state[:,stateIndex0] + blend * self.full_state[:,stateIndex1]
		# replace rotation angles with the quaternion blend
		#blended_state[3:6] = blend_root_rot

		return blended_state

	def get_state_at_time(self,time,env2D=False):
		""" Calculate the state at a desired time t. (calc_frame in motion_imitation) """
		f0, f1, blend = self.calc_blend_idx(time)
		# print('f0, f1, blend ', f0, f1, blend )
		blend_frame = self.blend_states(f0, f1, blend)
		if env2D:
			return blend_frame[indices2D]
		return blend_frame

	def get_torques_at_time(self,time,env2D=False):
		""" Calculate the torques at a desired time t. (calc_frame in motion_imitation) """
		f0, f1, blend = self.calc_blend_idx(time)
		blended_torques = (1.0 - blend) * self.taus[:,f0] + blend * self.taus[:,f1]
		return blended_torques

	def get_state_at_index(self,idx,env2D=False):
		""" Get the state at a desired index. """
		# if env2D:
		# 	return self.full_state2D[:,idx]
		if idx > self.trajLen - 1:
			return self.full_state[:,self.trajLen-1]
		return self.full_state[:,idx]
	
	def get_qDes_at_index(self, idx):
		if idx > self.trajLen - 1:
			return self.q[:,self.trajLen-1]
		return self.q[:,idx]
	
	def get_qdDes_at_index(self, idx):
		if idx > self.trajLen - 1:
			return self.dq[:,self.trajLen-1]
		return self.dq[:,idx]

	def get_torques_at_index(self,idx,env2D=False):
		""" Get the torques at a desired index. """
		# if env2D:
		# 	return self.full_state2D[:,idx]
		if idx > self.trajLen - 1:
			return self.taus[:,self.trajLen-1]
		return self.taus[:,idx]


	##############################################################################
	## helpers to extract base or joint info from a given full_state
	##############################################################################
	def get_base_pos_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns base position. 
		FULL: xyz 
		2D: x and z only
		"""
		# print("env2D:", env2D)
		if env2D:
			return state[self.BASE_POS_INDICES_2D]
		else:
			return state[self.BASE_POS_INDICES]



	def get_base_orn_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns base orientation. 
		FULL: RPY 
		2D: pitch only
		"""
		if env2D:
			return state[self.BASE_ORN_INDICES_2D]
		else:
			return state[self.BASE_ORN_INDICES]

	def get_base_linvel_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns base linear velocities. 
		FULL: dx, dy, dz 
		2D: dx and dz only
		"""
		if env2D:
			return state[self.BASE_LINVEL_INDICES_2D]
		else:
			return state[self.BASE_LINVEL_INDICES]

	def get_base_angvel_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns base angular velocities. 
		FULL: wx, wy, wz 
		2D: wy (pitch rate) only
		"""
		if env2D:
			return state[self.BASE_ANGVEL_INDICES_2D]
		else:
			return state[self.BASE_ANGVEL_INDICES]

	def get_joint_pos_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns joint positions. 
		FULL: (12) values 
		2D: (4) values, FR thigh/calf, RR thigh/calf
		"""
		if env2D:
			return state[self.JOINT_POS_INDICES_2D]
		else:
			return state[self.JOINT_POS_INDICES]

	def get_joint_vel_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns joint velocities. 
		FULL: (12) values 
		2D: (4) values, FR thigh/calf, RR thigh/calf
		"""
		if env2D:
			return state[self.JOINT_VEL_INDICES_2D]
		else:
			return state[self.JOINT_VEL_INDICES]

	##############################################################################
	## helpers to extract cartesian space info from a given full_state
	##############################################################################
	def get_foot_pos_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns leg frame foot positions. 
		FULL: (12) values 
		2D: TODO
		"""
		if env2D:
			raise ValueError('Not implemented: get_foot_pos_from_state() in 2D')
		else:
			return state[self.FOOT_POS_INDICES]

	def get_foot_vel_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns leg frame foot velocities. 
		FULL: (12) values 
		2D: TODO
		"""
		if env2D:
			raise ValueError('Not implemented: get_foot_vel_from_state() in 2D')
		else:
			return state[self.FOOT_VEL_INDICES]

	def get_foot_force_from_state(self,state,env2D=False):
		""" Takes in a full_state and returns leg frame foot forces.  
		FULL: (12) values 
		2D: TODO
		"""
		if env2D:
			raise ValueError('Not implemented: get_foot_force_from_state() in 2D')
		else:
			return state[self.FOOT_FORCE_INDICES]

