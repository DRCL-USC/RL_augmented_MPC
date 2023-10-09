"""This file implements the functionalities of a quadruped using pybullet.
getstate
send control
"""

from cmath import pi
import io
import os
import re
import math
import numpy as np
import os, inspect
import envs as envs
import matplotlib.pyplot as plt

# modify robot model urdf dir here
env_base_path = os.path.dirname(inspect.getfile(envs))
URDF_ROOT = os.path.join(env_base_path, 'assets/')
# URDF_FILENAME = "a1_description/urdf/a1_rm_fixhips_stl_v2.urdf"
URDF_FILENAME = "a1_description/urdf/a1.urdf"
_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")

ROBOT_MASS = 12
NUM_LEGS = 4
INIT_POSITION = [0, 0, 0.3]
INIT_ORIENTATION = [0, 0, 0, 1]
INIT_MOTOR_ANGLES = np.array([0, 0.4, -1.5] * NUM_LEGS)
TORQUE_LIMITS = np.asarray([33.5] * 12)

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ], dtype=np.float32)
SIDE_SIGN = [-1, 1, -1, 1]

LEG_OFFSETS = np.array([[0, -0.0838, 0], [0, 0.0838, 0], [0, -0.0838, 0], [0, 0.0838, 0]])

class A1(object):
    def __init__(self,
                 pybullet_client,
                 init_pos = [0, 0, 0.32],
                 init_ori = [0, 0, 0, 1],
                 time_step = 0.001):
        # self._config = robot_config
        self._num_motors = 12
        self._num_legs = 4
        self._pybullet_client = pybullet_client
        self._init_pos = init_pos
        self._init_ori = init_ori
        self.Kp_Joint = np.zeros(12, dtype=np.float32)
        self.Kd_Joint = np.zeros(12, dtype=np.float32)
        self.Kp_Cartesian = np.diag([0,0,0])
        self.Kd_Cartesian = np.diag([0,0,0])
        self.torque_cmds = np.zeros(12)

        self.q = np.zeros(12, dtype=np.float32)
        self.qdot = np.zeros(12, dtype=np.float32)
        self.position = np.zeros(3, dtype=np.float32)
        self.vWorld = np.zeros(3, dtype=np.float32)
        self.orientation = np.zeros(4, dtype=np.float32)
        self.R_body = np.zeros((3, 3), dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)
        self.omegaWorld = np.zeros(3, dtype=np.float32)
        self.foot_position_hip_frame = np.zeros((4, 3), dtype=np.float32)
        self.foot_position_base_frame = np.zeros((4, 3), dtype=np.float32)
        self.contacts = np.zeros(4, dtype=np.float32)

        # self.voltage_file = io.open('voltage.txt', 'w')

        self._LoadRobotURDF()
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._BuildUrdfMasses()
        self._RemoveDefaultJointDamping()
        self._BuildMotorIdList()
        self._SetMaxJointVelocities()

        self.Reset()

#================================= load urdf and setup ====================================#
    def _LoadRobotURDF(self):
        urdf_file = os.path.join(URDF_ROOT, URDF_FILENAME)
        self.A1 = self._pybullet_client.loadURDF(
            urdf_file,
            self._GetDefaultInitPosition(),
            self._GetDefaultInitOrientation(),
            flags = self._pybullet_client.URDF_USE_SELF_COLLISION
        )

        # self._pybullet_client.createConstraint(self.A1,  -1, -1, -1,
        #                                        self._pybullet_client.JOINT_FIXED, [0, 0, 0],
        #                                        [0, 0, 0], [0, 0, 1], childFrameOrientation=self._GetDefaultInitOrientation())

        return self.A1

    def _BuildJointNameToIdDict(self):
        """_BuildJointNameToIdDict """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
        ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        self._chassis_link_ids = [-1] # just base link
        self._leg_link_ids = []   # all leg links (hip, thigh, calf)
        self._motor_link_ids = [] # all leg links (hip, thigh, calf)

        self._joint_ids=[]      # all motor joints
        self._hip_ids = []      # hip joint indices only
        self._thigh_ids = []    # thigh joint indices only
        self._calf_ids = []     # calf joint indices only
        self._foot_link_ids = [] # foot joint indices

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            # print(joint_info)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if _CHASSIS_NAME_PATTERN.match(joint_name):
                self._chassis_link_ids = [joint_id]
            elif _HIP_NAME_PATTERN.match(joint_name):
                self._hip_ids.append(joint_id)
            elif _THIGH_NAME_PATTERN.match(joint_name):
                self._thigh_ids.append(joint_id)
            elif _CALF_NAME_PATTERN.match(joint_name):
                self._calf_ids.append(joint_id)
            elif _FOOT_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            else:
                continue
                raise ValueError("Unknown category of joint %s" % joint_name)

        # everything associated with the leg links
        self._joint_ids.extend(self._hip_ids)
        self._joint_ids.extend(self._thigh_ids)
        self._joint_ids.extend(self._calf_ids)
        # sort in case any weird order
        self._joint_ids.sort()
        self._hip_ids.sort()
        self._thigh_ids.sort()
        self._calf_ids.sort()
        self._foot_link_ids.sort()

        # print('joint ids', self._joint_ids)
        # sys.exit()

    def _BuildUrdfMasses(self):
        self._base_mass_urdf = []
        self._leg_masses_urdf = []
        self._foot_masses_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, chassis_id)[0])
        for leg_id in self._joint_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, leg_id)[0])
        for foot_id in self._foot_link_ids:
            self._foot_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, foot_id)[0])

    def _RemoveDefaultJointDamping(self):
        """Pybullet convention/necessity  """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            self._pybullet_client.changeDynamics(
                joint_info[0], -1, linearDamping=0, angularDamping=0)
        # for i in range(num_joints):
        #     self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.5)
        # set link friction (consistnet with Gazebo setup)
        for i in self._foot_link_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.6)
        for i in self._chassis_link_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._hip_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._thigh_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._calf_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)

    def _BuildMotorIdList(self):
    # self._motor_id_list = [self._joint_name_to_id[motor_name]
    #                         for motor_name in self._robot_config.MOTOR_NAMES]
        self._motor_id_list = self._joint_ids

    def _SetMaxJointVelocities(self):
        """Set maximum joint velocities from robot_config, the pybullet default is 100 rad/s """
        for i, link_id in enumerate(self._joint_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, maxJointVelocity=21.0)

    def _rand_helper_1d(self, orig_vec, percent_change):
        """Scale appropriately random in low/upp range, 1d vector """
        vec = np.zeros(len(orig_vec))
        for i, elem in enumerate(orig_vec):
            delta = percent_change * np.random.random() * orig_vec[i]
            vec[i] = orig_vec[i] + delta
        return vec

    def RandomizePhysicalParams(self):
        """Randomize physical robot parameters: masses. """
        base_mass = np.array(self._base_mass_urdf)
        leg_masses = np.array(self._leg_masses_urdf)
        foot_masses = np.array(self._foot_masses_urdf)

        new_base_mass = self._rand_helper_1d(base_mass, 0.8)
        new_leg_masses = self._rand_helper_1d(leg_masses, 0.5)
        new_foot_masses = self._rand_helper_1d(foot_masses, 0.5)

        self._pybullet_client.changeDynamics(self.A1, self._chassis_link_ids[0], mass=new_base_mass)
        for i, link_id in enumerate(self._joint_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, mass=new_leg_masses[i])
        for i, link_id in enumerate(self._foot_link_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, mass=new_foot_masses[i])

#================================= State feedback ====================================#
    def _GetDefaultInitPosition(self):
        return INIT_POSITION

    def _GetDefaultInitOrientation(self):
        return INIT_ORIENTATION

    def GetBasePosition(self):
        # in world frame
        position, _ = self._pybullet_client.getBasePositionAndOrientation(self.A1)
        self.position = np.asarray(position)

        #obtain local height similar to KF
        # body_height = []
        # stance_foot_ids = []

        # for i in range(4):
        #     if not self.contacts[i]:
        #         continue
        #     stance_foot_ids.append(i)
        #     body_height.append(-self.foot_position_hip_frame[i][2])

        # if len(body_height) != 0:
        #     self.position[2] = abs(sum(body_height)) / len(body_height)

        return self.position

    def GetBaseOrientation(self):
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.A1))
        self.orientation = np.asarray(orientation)
        return self.orientation

    def GetBaseRPY(self):
        # ori = self.GetBaseOrientation()
        self.rpy = np.asarray(self._pybullet_client.getEulerFromQuaternion(self.orientation))
        return self.rpy

    def GetBaseOrientationMatrix(self):
        """ Get the base orientation matrix, as numpy array """
        self.R_body = np.asarray(self._pybullet_client.getMatrixFromQuaternion(self.orientation)).reshape((3,3))
        return self.R_body

    def GetBaseLinearVelocity(self):
        """ Get base linear velocities (dx, dy, dz) in world frame"""
        vWorld, _ = np.asarray(self._pybullet_client.getBaseVelocity(self.A1))
        self.vWorld = np.asarray(vWorld)
        return self.vWorld

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                       orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def GetBaseAngularVelocity(self):
        """ Get base angular velocities (droll, dpitch, dyaw) in world frame"""
        _, angVel = self._pybullet_client.getBaseVelocity(self.A1)
        self.omegaWorld = np.asarray(angVel)
        return self.omegaWorld

    def GetBaseAngularVelocityLocalFrame(self):
        _, angVel = self._pybullet_client.getBaseVelocity(self.A1)
        # orientation = self.GetBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angVel, self.orientation)

    def GetMotorAngles(self):
        """Get all motor angles """
        motor_angles = [
            self._pybullet_client.getJointState(self.A1, motor_id)[0]
            for motor_id in self._motor_id_list
        ]
        self.q = np.asarray(motor_angles)
        return motor_angles

    def GetMotorVelocities(self):
        """Get the velocity of all motors."""
        motor_velocities = [
            self._pybullet_client.getJointState(self.A1, motor_id)[1]
            for motor_id in self._motor_id_list
        ]

        self.qdot = np.asarray(motor_velocities)
        return motor_velocities

    def GetMotorTorqueCmds(self):
        return self.torque_cmds

    def GetHipPositionsInBaseFrame(self):
        return HIP_OFFSETS

    # def GetHipOffsetsInBaseFrame(self):
    #     return (HIP_OFFSETS + LEG_OFFSETS)

    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.A1)

        self.contacts = [False, False, False, False]
        for contact in all_contacts:
            if contact[2] == self.A1:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[3])
                self.contacts[toe_link_index] = True
            except ValueError:
                continue

        return self.contacts

    def GetInvalidContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.A1)
        for c in all_contacts:
            if c[3] in self._thigh_ids or c[4] in self._thigh_ids:
                return True
            if c[3] in self._calf_ids or c[4] in self._calf_ids:
                return True
            # if c[3] in self._hip_ids or c[4] in self._hip_ids:
            #     return True
            # if c[3] in self._chassis_link_ids or c[4] in self._chassis_link_ids:
            #     return True

        return False

    def get_cam_view(self):
        original_cam_look_direction = np.array([1, 0, 0])  # Same as original robot orientation

        pos = self.position
        orientation = self.orientation
        axis, ori = self._pybullet_client.getAxisAngleFromQuaternion(orientation)
        axis = np.array(axis)

        original_cam_up_vector = np.array([0, 0, 1])  # Original camera up vector

        new_cam_up_vector = np.cos(ori) * original_cam_up_vector + np.sin(ori) * np.cross(axis, original_cam_up_vector) + (
                1 - np.cos(ori)) * np.dot(axis, original_cam_up_vector) * axis  # New camera up vector

        new_cam_look_direction = np.cos(ori) * original_cam_look_direction + np.sin(ori) * np.cross(axis, original_cam_look_direction) + (
                1 - np.cos(ori)) * np.dot(axis, original_cam_look_direction) * axis  # New camera look direction

        new_target_pos = pos + new_cam_look_direction  # New target position for camera to look at

        new_cam_pos = pos + 0.28 * new_cam_look_direction

        viewMatrix = self._pybullet_client.computeViewMatrix(
        cameraEyePosition=new_cam_pos,
        cameraTargetPosition=new_target_pos,
        cameraUpVector=new_cam_up_vector)

        near = 0.01
        far = 1000

        projectionMatrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=87.0,
        aspect=1.0,
        nearVal=near,
        farVal=far)

        _, _, _, depth_buffer, _ = self._pybullet_client.getCameraImage(
        width=32,
        height=32,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

        depthImg =  far * near / (far - (far - near) * depth_buffer)
        depthImg = np.clip(depthImg, a_min=0.2, a_max=10)

        return depthImg

#================================== Send Cmd =======================================#
    def _setMotorTorqueById(self, motor_id, torque):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self.A1,
                                                    jointIndex=motor_id,
                                                    controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                    force=torque)

    def ApplyAction(self, kpJoint, kdJoint, qDes, qdotDes, tauDes):
        motor_torque = kpJoint * (qDes - self.q) + kdJoint * (qdotDes - self.qdot) + tauDes
        self.torque_cmds = self.ApplyMotorDynamicsConstraint(self.qdot, motor_torque)

        for motor_id, torque in zip(self._motor_id_list, self.torque_cmds):
            self._setMotorTorqueById(motor_id, torque)

        # update states
        self.GetMotorAngles()
        self.GetMotorVelocities()
        self.GetFootContacts()
        self.GetBaseOrientation()
        self.GetBaseAngularVelocity()
        self.GetBaseOrientationMatrix()
        self.GetBaseRPY()
        self.GetFootPositionsInBaseFrame()
        self.GetBaseLinearVelocity()
        self.GetBasePosition()


    def SetCartesianPD(self, kpCartesian, kdCartesian):
        self.Kp_Cartesian = kpCartesian
        self.Kd_Cartesian = kdCartesian
    
    def ComputeLegImpedanceControl(self, pDes, vDes, legID):
        # q = self.GetMotorAngles()
        # qdot = self.GetMotorVelocities()
        torque = np.zeros(3)

        pFoot = self.ComputeFootPosHipFrame(self.q[legID*3: legID*3+3], legID)
        J = self.ComputeLegJacobian(self.q[legID*3: legID*3+3], legID)
        vFoot = self.ComputeFootVelHipFrame(self.q[legID*3: legID*3+3], self.qdot[legID*3: legID*3+3], legID)

        torque = J.T @ (self.Kp_Cartesian@(pDes - pFoot) \
                          + self.Kd_Cartesian@(vDes - vFoot))

        return torque


    def ComputeImpedanceControl(self, pDes, vDes):
        torque = np.zeros(12)
        for i in range(4):
            pFoot = self.ComputeFootPosHipFrame(self.q[i*3: i*3+3], i)
            J = self.ComputeLegJacobian(self.q[i*3: i*3+3], i)
            vFoot = self.ComputeFootVelHipFrame(self.q[i*3: i*3+3], self.qdot[i*3: i*3+3], i)

            torque[i*3:i*3+3] = J.T @ (self.Kp_Cartesian@(pDes[i*3:i*3+3] - pFoot) \
                          + self.Kd_Cartesian@(vDes[i*3:i*3+3] - vFoot))

        return torque

    def ComputeForceControl(self, force_cmd):
        # q = self.GetMotorAngles()
        torque = np.zeros(12)
        for i in range(4):
            J = self.ComputeLegJacobian(self.q[i*3:i*3+3], i)
            torque[i*3:i*3+3] = force_cmd[i*3:i*3+3] @ J    # J^T times F, but F is (1x3). A.T * B.T = (AB).T = BA

        return torque

    def ApplyMotorDynamicsConstraint(self, motor_velocity, motor_torques):
        # Kt = 4 / 34  # from Unitree
        # self._voltage_max = 24
        # self._gear_ratio = 9.1
        # self._R_motor = 0.346     

        # voltage = np.zeros(12)

        # for i in range(12):
        #     voltage[i] = motor_torques[i] * self._R_motor / (self._gear_ratio*Kt) + motor_velocity[i] * self._gear_ratio * Kt

        #     # self.voltage_file.write( "%.5f " %voltage[i])
        #     # print("voltage", i, ":", voltage[i])
        #     if voltage[i] > self._voltage_max:
        #         motor_torques[i] = (self._voltage_max-motor_velocity[i]*self._gear_ratio*Kt)*(self._gear_ratio*Kt/self._R_motor)
        #     if voltage[i] <- self._voltage_max:
        #         motor_torques[i] = (-self._voltage_max-motor_velocity[i]*self._gear_ratio*Kt)*(self._gear_ratio*Kt/self._R_motor)

        # self.voltage_file.write("\n")
        return np.clip(motor_torques, -33.5, 33.5)
        


#=================================== Kinematics & motor dynamics constraint ========================================#
    def q1_ik(self, py, pz, l1):
        L = np.sqrt(py**2 + pz**2 - l1**2)
        q1 = np.arctan2(pz*l1 + py*L, py*l1-pz*L)

        return q1
    
    def q2_ik(self, q1, q3, px, py, pz, b3z, b4z):
        a1 = py*np.sin(q1) - pz*np.cos(q1)
        a2 = px
        m1 = b4z*np.sin(q3)
        m2 = b3z + b4z*np.cos(q3)
        q2 = np.arctan2(m1*a1 + m2*a2, m1*a2 - m2*a1)

        return q2
    
    def q3_ik(self, b3z, b4z, b):
        temp = (b3z**2 + b4z**2 - b**2)/(2*abs(b3z*b4z))
        temp = np.clip(temp, -1, 1)
        q3 = np.arccos(temp)
        q3 = -(np.pi - q3)

        return q3
        
    def ComputeLegIK(self, foot_position, leg_id):

        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        b2y = l1 * SIDE_SIGN[leg_id]
        b3z = -l2
        b4z = -l3
        a = l1
        c = np.sqrt(foot_position[0]**2 + foot_position[1]**2 + foot_position[2]**2)
        b = np.sqrt(c**2 - a**2)

        q1 = self.q1_ik(foot_position[1], foot_position[2], b2y)
        q3 = self.q3_ik(b3z, b4z, b)
        q2 = self.q2_ik(q1, q3, foot_position[0], foot_position[1], foot_position[2], b3z, b4z)

        qDes = np.array([q1, q2, q3])
        if np.isnan(qDes).any():
            qDes = np.array([0, 0.5, -1.4])

        return qDes

    def ComputeFootPosHipFrame(self, q, leg_id):

        side_sign = 1
        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        pos = np.zeros(3)
        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * side_sign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * side_sign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        return pos

    def ComputeLegJacobian(self, q, leg_id):
        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        J = np.zeros((3,3))
        J[1, 0] = -SIDE_SIGN[leg_id] * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
        J[2, 0] = SIDE_SIGN[leg_id] * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
        J[0, 1] = -l3 * c23 - l2 * c2
        J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
        J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
        J[0, 2] = -l3 * c23
        J[1, 2] = -l3 * s23 *s1
        J[2, 2] = l3 * s23 * c1

        return J

    def ComputeFootVelHipFrame(self, q, qdot, leg_id):
        J = self.ComputeLegJacobian(q, leg_id)
        foot_vel = J @ qdot
        return foot_vel

    def GetFootPositionsInBaseFrame(self):
        joint_angles = self.q.reshape((4,3))
        # self.foot_positions = np.zeros((4,3))
        for i in range(4):
            self.foot_position_hip_frame[i] = self.ComputeFootPosHipFrame(joint_angles[i], i)
        self.foot_position_base_frame = self.foot_position_hip_frame + HIP_OFFSETS
        return self.foot_position_base_frame
    


    #==================================== Reset =======================================#
    def Reset(self):
        self._pybullet_client.resetBasePositionAndOrientation(self.A1,
                                                              self._GetDefaultInitPosition(),
                                                              self._GetDefaultInitOrientation())
        self._pybullet_client.resetBaseVelocity(self.A1, [0, 0, 0], [0, 0, 0])
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.A1,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for i, jointId in enumerate(self._joint_ids):
            angle = INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(
                self.A1, jointId, angle, targetVelocity=0
            )

        self.GetMotorAngles()
        self.GetMotorVelocities()

    def Reset_to_position(self, y = 0):
        self._pybullet_client.resetBasePositionAndOrientation(self.A1,
                                                              [0, y ,0.3],
                                                              self._GetDefaultInitOrientation())
        self._pybullet_client.resetBaseVelocity(self.A1, [0, 0, 0], [0, 0, 0])
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.A1,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for i, jointId in enumerate(self._joint_ids):
            angle = INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(
                self.A1, jointId, angle, targetVelocity=0
            )

        self.GetMotorAngles()
        self.GetMotorVelocities()
