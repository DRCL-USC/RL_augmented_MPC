from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import inspect
import math
import os
import io
import random
import time
from collections import deque
import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import scipy.interpolate
from absl import app
from absl import flags
from gym import spaces
from mpc_implementation import MPClocomotion

import envs.A1 as a1
import envs.assets as assets

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.05

INIT_MOTOR_ANGLES = np.array([0, 0.5, -1.4] * 4)
EPISODE_LENGTH = 15

class QuadrupedGymEnv(gym.Env):
    def __init__(
        self,
        time_step,
        action_repeat,
        obs_hist_len,
        render = False,
        **kwargs):

        self._action_repeat = action_repeat
        self._render = render
        self._action_bound = 1.0
        self._time_step = time_step
        self._num_bullet_solver_iterations = 60
        self._obs_hist_len = obs_hist_len
        self._MAX_EP_LEN = EPISODE_LENGTH
        self._urdf_root = assets.getDataPath()
        # self._last_cmd = np.zeros(14)
        self._last_cmd = np.zeros(18)

        # self.num_obs = 78
        self.num_obs = 82

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_frame_time = 0.0
        self._terminate = False
        self.base_block_ID = -1

        if self._render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
            self.q_file = io.open('v_robot.txt', 'w')
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root,
                                                  basePosition=[80,0,0],
                                                  )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1)
        self._pybullet_client.setRealTimeSimulation(0)

        self._robot = a1.A1(pybullet_client = self._pybullet_client)
        # self._robot = aliengo.Aliengo(pybullet_client = self._pybullet_client)
        self.vxCommand = 0
        self.vyCommand = 0
        self.yaw_rate_cmd = 0

        self.vel_reward = 0
        self.height_reward = 0

        self.x_world = 0
        self.y_world = 0

        self.box_ids = []

        self.setupActionSpace()
        self.setupObservationSpace()

        self.controller = MPClocomotion.MPCLocomotion(0.001, 30)


    def setupActionSpace(self):
        # action_dim = 14
        action_dim = 18
        self._action_dim = action_dim
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high - ACTION_EPS, action_high + ACTION_EPS, dtype=np.float32)
        self._last_action_rl = np.zeros(self._action_dim)
    
    def setupObservationSpace(self):
        upper_bound = np.array([50] * self.num_obs * self._obs_hist_len)
        lower_bound = np.array([-50] * self.num_obs * self._obs_hist_len)

        self.observation_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)
    
    def reset(self):
        self._robot.Reset()
        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._terminate = False
        self._height = 0.3
        self.z_force = 0
        self.x_force = 0
        self.y_force = 0
        self.x_tau = 0
        self.y_tau = 0
        self.z_tau = 0
        # self._last_cmd = np.zeros(14)
        self._last_cmd = np.zeros(18)

        self.vel_reward = 0
        self.height_reward = 0

        self.x_world = 0
        self.y_world = 0

        # self.add_random_boxes()
        # if self.base_block_ID != -1:
        #     self._pybullet_client.removeBody(self.base_block_ID)
        #     self.base_block_ID = -1
        # if np.random.random() < 0.8:
        #     self.add_base_mass_offset()

        # self.vxCommand = -0.5 + 1.0 * np.random.random()
        # self.vyCommand = -0.5 + np.random.random()
        # self.yaw_rate_cmd = 0
        # -1 + 2.0 * np.random.random()

        self.controller.initialize()
        # self.controller.setupCmd(0.0, 0.0, 0.0, self._height)
        if self.render:
            if self.base_block_ID != -1:
                self._pybullet_client.removeBody(self.base_block_ID)
                self.base_block_ID = -1
            # if np.random.random() < 0.8:
            self.add_base_mass_offset(spec_mass=5.0)
            self._settle_robot()

        self._obs_buffer = deque([np.zeros(self.observation_space.shape[0])] * self._obs_hist_len)
        for _ in range(self._obs_hist_len):
            self.getObservation()

        # self._last_action_rl = np.zeros(8)
        self._last_action_rl = np.zeros(12)

        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
        
        return self.getObservation()

    def _settle_robot(self):
        kp_joint = np.array([60]*12)
        kd_joint = np.array([5]*12)
        pDes = np.array([0, -0.0838, -0.3, 0, 0.0838, -0.3, -0.0, -0.0838, -0.3, -0.0, 0.0838, -0.3])
        for _ in range(2000):
            qDes = np.zeros(12)
            for i in range(4):
                qDes[i*3:i*3+3] = self._robot.ComputeLegIK(pDes[i*3:i*3+3], i)
            self._robot.ApplyAction(kp_joint, kd_joint, qDes, np.zeros(12), np.zeros(12))
            if self._render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()
    
    def getObservation(self): # dummy function to test MPC
        observation = []

        observation.extend(list(self._robot.rpy))
        observation.extend(list(self._robot.vWorld))# use world frame for now
        observation.extend(list(self._robot.omegaWorld))
        observation.extend(self._robot.q)
        observation.extend(self._robot.qdot)
        # observation.extend(self._robot.foot_position_base_frame[0].tolist())
        # observation.extend(self._robot.foot_position_base_frame[1].tolist())
        # observation.extend(self._robot.foot_position_base_frame[2].tolist())
        # observation.extend(self._robot.foot_position_base_frame[3].tolist())
        observation.extend(list(self.controller.foothold_heuristic.reshape((8))))
        observation.extend(list(self._robot.contacts))
        observation.extend(list(self.controller.contactState))
        observation.extend(list(self.controller.f_ff.reshape((12))/12.0))
        robot_yaw = self._robot.rpy[2]
        vx = np.cos(robot_yaw) * self.vxCommand - np.sin(robot_yaw) * self.vyCommand
        vy = np.sin(robot_yaw) * self.vxCommand + np.cos(robot_yaw) * self.vyCommand
        # yaw_rate = self.yaw_rate_cmd
        # height = self._height
        # pitch_cmd = 0.0
        # roll_cmd = 0.0
        observation.extend(np.array([vx, vy, self.yaw_rate_cmd]))
        observation.extend(list(self._last_cmd))

        self._obs_buffer.appendleft(observation)
        obs = []
        for i in range(self._obs_hist_len):
            obs.extend(self._obs_buffer[i])
        return obs
    
    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1)*(upper_lim - lower_lim)
        a = np.clip(a, lower_lim, upper_lim)
        return a

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)

        # if self._env_step_counter % 50 == 0 and np.random.random() > 0.8:
        #     self.vxCommand = -1 + 2.0 * np.random.random()
        #     self.vyCommand = -0.5 + np.random.random()
        #     self.yaw_rate_cmd = -2 + 4.0 * np.random.random()
        
        # if self._sim_step_counter < 5000:
        self.vxCommand = 0.5
        # self.vyCommand = 0.3
        # self.yaw_rate_cmd = - 1
        # if np.random.random() > 0.7:
        #     self.z_force = 0
        # else:
        #     self.z_force = - 80 * np.random.random()
        # if self._env_step_counter % 10 == 0 and np.random.random() > 0.5:
        #     self.x_force = -20 + 40 * np.random.random()
        #     self.y_force = -20 + 40 * np.random.random() 
        #     self.x_tau = self.z_force * (-0.1 + 0.2 * np.random.random()) - 2 + 4* np.random.random()
        #     self.y_tau = self.z_force * (-0.1 + 0.2 * np.random.random()) - 3 + 6 * np.random.random()
        #     self.z_tau = -1 + 2 * np.random.random()

        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        offsets = self._get_desried_accel(action)
        self.controller.setupCmd(self.vxCommand, self.vyCommand, self.yaw_rate_cmd, self._height)
        accel_offset = offsets[0:6]
        # foothold_offset = offsets[6:]
        self.controller.setDesiredAccel(accel_offset)
        # self.controller.setFootholdOffset(foothold_offset)
        joint_angle_offset = offsets[6:]

        for _ in range(self._action_repeat):
            self.controller.run(self._robot)
            tau = self._robot.ComputeForceControl(self.controller.f_ff.reshape((12)))
                    
            Jointkp = np.zeros(12)
            Jointkd = np.zeros(12)
            qDes = np.zeros(12)
                
            for i in range(4):
                if self.controller.contactState[i] == 0:
                    # self._robot.SetCartesianPD(np.diag([450, 450, 250]), np.diag([10, 10, 10]))
                    # tau[i*3:i*3+3] += self._robot.ComputeLegImpedanceControl(self.controller.p_des_leg[i], self.controller.v_des_leg[i], i)
                    qDes[i*3:i*3+3] = self._robot.ComputeLegIK(self.controller.p_des_leg[i], i) + joint_angle_offset[i * 3:i * 3 + 3]
                    Jointkp[i*3:i*3+3] = np.array([60, 60, 60])
                    Jointkd[i*3:i*3+3] = np.array([1, 1, 1])
                else:
                    Jointkp[i*3:i*3+3] = np.array([0, 0, 0])
                    Jointkd[i*3:i*3+3] = np.array([0.5, 0.5, 0.5])

            self._robot.ApplyAction(Jointkp, Jointkd, qDes, np.zeros(12), tau)

            self._pybullet_client.applyExternalForce(self._robot.A1, -1, (self.x_force, self.y_force, self.z_force), (0,0,0), self._pybullet_client.LINK_FRAME)
            self._pybullet_client.applyExternalTorque(self._robot.A1, -1, (self.x_tau, self.y_tau, 0.0), self._pybullet_client.LINK_FRAME)
                
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self._robot.q)
            self._dt_motor_velocities.append(self._robot.qdot)

            if self._render:
                    # self.q_file.write( "%.5f " %self._robot.GetBaseLinearVelocity()[0])
                    # self.q_file.write( "%.5f " %self._robot.GetBaseLinearVelocity()[1])
                    # self.q_file.write( "%.5f " %self._robot.GetBaseLinearVelocity()[2])
                    # self.q_file.write( "%.5f " %accel_offset[3])
                    # self.q_file.write( "%.5f " %accel_offset[4])
                    # self.q_file.write( "%.5f \n" %accel_offset[5])
                self._render_step_helper()

        self._last_action_rl = action
        self._last_cmd = offsets
        self._env_step_counter += 1
        done = False
        reward = self.get_reward()

        self.x_world = self._robot.position[0]
        self.y_world = self._robot.position[1]

        if self.termination():
            done = True
            reward -= 20
        if self.get_sim_time() > self._MAX_EP_LEN:
            done = True
            # reward += 2
        # if done:
        #     print("ori_reward: ", self.orn_reward, " height reward: ", self.height_reward)
        return np.array(self.getObservation()), reward, done, {'base_pos': self._robot.position,
                                                               'base_vel': self._robot.vWorld,
                                                               'action': self._last_cmd}
    
    def _get_desried_accel(self, action):
        # ub_foot_pos_offset = np.array([0.1, -0.1, 0.1, 0.1]*2)
        # lb_foot_pos_offest = np.array([-0.1, 0.1, -0.1, -0.1]*2)
        # ub_desired_accel = np.array([0.4, 0.4, 0.0]) * 9.8
        # lb_desired_accel = np.array([-0.4, -0.4, -0.8]) * 9.8
        # ub_desired_accel = np.array([ 5.0,  10.0,  2.0,  4.0,  2.0,  2.0,  0.1,   0.05,  0.1,   0.05,  0.05,  0.05 ,  0.05,  0.05]) # 10Nm
        # lb_desired_accel = np.array([-5.0, -10.0, -2.0, -4.0, -2.0, -8.0, -0.05, -0.05, -0.05, -0.05, -0.1, -0.05 ,  -0.1, -0.05])
        # ub_desired_accel = np.array([ 5.0,  10.0,  2.0,  4.0,  2.0,  2.0,  0.1,   0.05,  0.1,   0.05,  0.05,  0.05 ,  0.05,  0.05]) # 10Nm
        # lb_desired_accel = np.array([-5.0, -10.0, -2.0, -4.0, -2.0, -8.0, -0.05, -0.05, -0.05, -0.05, -0.1, -0.05 ,  -0.1, -0.05])
            # ub_desired_accel = np.array([ 5.0,  10.0,  2.0,  4.0,  2.0,  2.0,  0.4,   0.4,  0.4,   0.4,  0.4,  0.4 ,  0.4,  0.4, 0.4, 0.4, 0.4, 0.4]) # 10Nm
        # lb_desired_accel = np.array([-5.0, -10.0, -2.0, -4.0, -2.0, -8.0, -0.4, -0.4, -0.4, -0.4, -0.4, -0.4 ,  -0.4, -0.4, -0.4, -0.4 ,-0.4 ,-0.4])
        ub_desired_accel = np.array([ 5.0,  10.0,  2.0,  4.0,  2.0,  2.0,  0.2,   0.2,  0.2, 0.2,  0.2,  0.2 ,  0.2,  0.2, 0.2, 0.2, 0.2, 0.2]) # 10Nm
        lb_desired_accel = np.array([-5.0, -10.0, -2.0, -4.0, -2.0, -8.0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2 ,  -0.2, -0.2, -0.2, -0.2 ,-0.2 ,-0.2])
        accel_offset = self._scale_helper(action, lb_desired_accel, ub_desired_accel)
        # accel_offset *= 9.8
    
        # return np.array([0.0, 0.0, 0.0, accel_offset[0], accel_offset[1], accel_offset[2]])
        return accel_offset

    def get_reward(self):
        survival_reward = 0.002
        com_pos = self._robot.position
        yaw_dot = self._robot.omegaWorld[2]
        robot_yaw = self._robot.rpy[2]
        # change to average velocity
        avg_vx = (com_pos[0] - self.x_world)/0.03
        avg_vy = (com_pos[1] - self.y_world)/0.03
        energy_reward = 0

        for tau, vel in zip(self._dt_motor_torques, self._dt_motor_velocities):
            energy_reward -= 0.005 * np.abs(np.dot(tau, vel)) * self._time_step

        des_vel_x_world_frame = np.cos(robot_yaw) * self.vxCommand - np.sin(robot_yaw) * self.vyCommand
        des_vel_y_world_frame = np.sin(robot_yaw) * self.vxCommand + np.cos(robot_yaw) * self.vyCommand 
        vel_reward = 0.08 * (0.05 - abs(avg_vx - des_vel_x_world_frame)) + 0.08*(0.05 - abs(avg_vy - des_vel_y_world_frame)) \
                     - 0.04 * abs(yaw_dot - self.yaw_rate_cmd)

        height_reward = 0.3 * (0.02 - abs(self._robot.position[2] - 0.3))
        # height_reward = - abs(self._robot.GetBasePosition()[2] - 0.3)
        orn_reward = 0.02 * (0.05 - abs(self._robot.rpy[0]) + 0.05 - abs(self._robot.rpy[1]))

        self.vel_reward += vel_reward
        self.height_reward += height_reward

        # self.height_reward += height_reward

        return vel_reward + height_reward + survival_reward + orn_reward + energy_reward

    def get_sim_time(self):
        return self._sim_step_counter * self._time_step

    def termination(self):
        rpy = self._robot.rpy
        pos = self._robot.position

        # return self.is_fallen() or distance > self._distance_limit #or numInvalidContacts > 0
        return abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0 or pos[2] < 0.12 
                # or self._robot.GetInvalidContacts())

    def scale_rand(self, num_rand, low, high):
        """ scale number of rand numbers between low and high """
        return low + np.random.random(num_rand) * (high - low)

    def add_base_mass_offset(self, spec_mass=None, spec_location=None):
        quad_base = np.array(self._robot.position)
        quad_ID = self._robot.A1
        # quad_ID = self._robot.Aliengo

        # offset_low = np.array([-0.15, -0.02, -0.05])
        # offset_upp = np.array([ 0.15,  0.02,  0.05])
        offset_low = np.array([-0.01, -0.01, 0.0])
        offset_upp = np.array([0.01, 0.01, 0.05])
        #   block_pos_delta_base_frame = -1*np.array([-0.2, 0.1, -0.])
        if spec_location is None:
            block_pos_delta_base_frame = np.array([0.05, 0.0, 0.05])
        else:
            block_pos_delta_base_frame = np.array(spec_location)
        if spec_mass is None:
            # base_mass = 8*np.random.random()
            # base_mass = 15*np.random.random()
            # base_mass = 12*np.random.random()
            base_mass = 8 * np.random.random()
        else:
            base_mass = spec_mass
        if self._render:
            print('=========================== Random Mass:')
            print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)

            # if rendering, also want to set the halfExtents accordingly
            # 1 kg water is 0.001 cubic meters
        # boxSizeHalf = [(base_mass * 0.001) ** (1 / 3) / 2] * 3
            # boxSizeHalf = [0.05]*3
        translationalOffset = [0, 0, 0.05]
        # else:
        boxSizeHalf = [(6.0 * 0.001) ** (1 / 3) / 2] * 3 # change size later
        #     translationalOffset = [0] * 3

        # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=[0.05]*3)
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf,
                                                               collisionFramePosition=translationalOffset)
        # orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        self.base_block_ID = self._pybullet_client.createMultiBody(baseMass=base_mass,
                                                                   baseCollisionShapeIndex=sh_colBox,
                                                                   basePosition=quad_base + block_pos_delta_base_frame,
                                                                   baseOrientation=[0, 0, 0, 1])

        cid = self._pybullet_client.createConstraint(quad_ID, -1, self.base_block_ID, -1,
                                                     self._pybullet_client.JOINT_FIXED,
                                                     [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
        # disable self collision between box and each link
        for i in range(-1, self._pybullet_client.getNumJoints(quad_ID)):
            self._pybullet_client.setCollisionFilterPair(quad_ID, self.base_block_ID, i, -1, 0)

    def add_random_boxes(self, num_rand=60):
        """Add random boxes in front of the robot, should be in x [0.5, 50] and y [-5,5]
    how many?
    how large?
    """
        # print('-'*80,'\nadding boxes\n','-'*80)
        # x location
        x_low = 0.5
        x_upp = 5
        # y location
        y_low = -5
        y_upp = 5
        # z, orig [0.01, 0.03]
        z_low = 0.03  #
        z_upp = 0.07  # max height, was 0.025
        # z_upp = 0.04 # max height, was 0.025
        # block dimensions
        block_x_max = 1
        block_x_min = 0.1
        block_y_max = 1
        block_y_min = 0.1
        # block orientations
        # roll_low, roll_upp = -0.1, 0.1
        # pitch_low, pitch_upp = -0.1, 0.1
        roll_low, roll_upp = -0.01, 0.01
        pitch_low, pitch_upp = -0.01, 0.01  # was 0.001,
        yaw_low, yaw_upp = -np.pi, np.pi

        x = self.scale_rand(num_rand, x_low, x_upp)
        y = self.scale_rand(num_rand, y_low, y_upp)
        z = self.scale_rand(num_rand, z_low, z_upp)
        block_x = self.scale_rand(num_rand, block_x_min, block_x_max)
        block_y = self.scale_rand(num_rand, block_y_min, block_y_max)
        roll = self.scale_rand(num_rand, roll_low, roll_upp)
        pitch = self.scale_rand(num_rand, pitch_low, pitch_upp)
        yaw = self.scale_rand(num_rand, yaw_low, yaw_upp)
        # loop through
        if not self.box_ids:
            for i in range(num_rand):
                sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                       halfExtents=[block_x[i] / 2, block_y[i] / 2,
                                                                                    z[i] / 2])
                orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
                if abs(x[i]) < 0.5 and abs(y[i]) < 0.5: # prevent obstacle generated at robot's location
                    x[i] += 1
                    y[i] += 1
                
                block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                               basePosition=[x[i], y[i], z[i] / 2], baseOrientation=orn)
                # set friction coeff to 1
                self._pybullet_client.changeDynamics(block2, -1, lateralFriction=1)
                self.box_ids.append(block2)

            # add walls
            # orn = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
            # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            #                                                        halfExtents=[x_upp / 2, 0.5, 0.5])
            # block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
            #                                                basePosition=[x_upp / 2, y_low, 0.5], baseOrientation=orn)
            # block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
            #                                                basePosition=[x_upp / 2, -y_low, 0.5], baseOrientation=orn)
        else:
            for i in range(num_rand):
                orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
                if abs(x[i]) < 0.5 and abs(y[i]) < 0.5: # prevent obstacle generated at robot's location
                    x[i] += 1
                    y[i] += 1
                self._pybullet_client.resetBasePositionAndOrientation(self.box_ids[i],
                                                                      posObj=[x[i], y[i], z[i] / 2], ornObj=orn)
#========================================= render ======================================#
    def close(self):
        self._pybullet_client.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render_step_helper(self):
        """ Helper to configure the visualizer camera during step(). """
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self._last_frame_time
        # print('time_spent ', time_spent)
        self._last_frame_time = time.time()
        # time_to_sleep = self._action_repeat * self._time_step - time_spent
        time_to_sleep = self._time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._time_step):
            time.sleep(time_to_sleep)
        base_pos = self._robot.GetBasePosition()
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [
            base_pos[0], base_pos[1],
            0.3
        ]
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)
    
    def _configure_visualizer(self):

        self._render_width = 960
        self._render_height = 720
        self._cam_dist = 1.5 # .75 #for paper
        self._cam_yaw = 20
        self._cam_pitch = -20 # -10 # for paper
        # get rid of random visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)
# 
    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = np.array([self._robot.GetBasePosition()[0],self._robot.GetBasePosition()[1], 0.3])
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                    aspect=float(self._render_width)/self._render_height,
                                                                    nearVal=0.1,
                                                                    farVal=100.0)
        (_, _, px, _,
        _) = self._pybullet_client.getCameraImage(width=self._render_width,
                                                height=self._render_height,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
