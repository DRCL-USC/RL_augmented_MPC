import os, ray, sys, time, io
import numpy as np
import pickle as pickle
import gym
import json
# from json import JSONEncoder
from ray import tune
from ray.rllib import SampleBatch
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.tune.trial import ExportFormat
# from envs.pmtg_task import QuadrupedGymEnv

# from envs.running_task_cartesian import QuadrupedGymEnv
# from envs.running_task_joint import QuadrupedGymEnv
# from envs.MPC_task import QuadrupedGymEnv
# from envs.MPC_foot_reaction_env import  QuadrupedGymEnv
from envs.MPC_task_accel_foothold import QuadrupedGymEnv
# from envs.loco_manip_task import QuadrupedManipEnv
# from envs.jump_task_2D import QuadrupedGymEnv
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ray.rllib.models import ModelCatalog

from learning.rllib_helpers.fcnet_me import MyFullyConnectedNetwork
# from learning.rllib_helpers.tf_lstm_net import LSTMModel

from matplotlib import pyplot as plt

# model_dir = "ray_results_ppo/Jump2D-1121-16-29/PPO/PPO_DummyQuadrupedEnv_b8c7f_00000_0_2022-11-21_16-29-27" # 0.8
# model_dir = "ray_results_ppo/Jump2D-1121-23-18/PPO/PPO_DummyQuadrupedEnv_deec9_00000_0_2022-11-21_23-18-32" # 1
# model_dir = "ray_results_ppo/Jump2D-1123-01-55/PPO/PPO_DummyQuadrupedEnv_0894e_00000_0_2022-11-23_01-56-00" # 0.4
# model_dir = "ray_results_ppo/Jump2D-1123-10-35/PPO/PPO_DummyQuadrupedEnv_a142d_00000_0_2022-11-23_10-35-40" #0.2
# model_dir = "ray_results_ppo/Jump2D-1130-16-58/PPO/PPO_DummyQuadrupedEnv_41a98_00000_0_2022-11-30_16-58-23" # 0.8 with motor dynamics
# model_dir = "ray_results_ppo/Jump2D-1201-11-38/PPO/PPO_DummyQuadrupedEnv_b9095_00000_0_2022-12-01_11-38-25" #0.4 with motor dynamics
# model_dir = "ray_results_ppo/Jump2D-1208-17-44/PPO/PPO_DummyQuadrupedEnv_0f628_00000_0_2022-12-08_17-44-46" #0.8m new energy reward
# model_dir = "ray_results_ppo/Jump2D-1209-03-20/PPO/PPO_DummyQuadrupedEnv_6f2e1_00000_0_2022-12-09_03-20-06" # 0.8 new energy reward
# model_dir = "ray_results_ppo/Jump2D-1209-22-24/PPO/PPO_DummyQuadrupedEnv_612c3_00000_0_2022-12-09_22-25-02" # 0.7, small friction
# model_dir = "ray_results_ppo/Jump2D-1210-19-04/PPO/PPO_DummyQuadrupedEnv_8d040_00000_0_2022-12-10_19-04-39" # 0.7, large friction
# model_dir = "ray_results_ppo/Jump2D-1212-19-55/PPO/PPO_DummyQuadrupedEnv_ec337_00000_0_2022-12-12_19-55-03" # 0.8 100Hz no energy penalty
# model_dir = "ray_results_ppo/Jump2D-1212-20-32/PPO/PPO_DummyQuadrupedEnv_30d69_00000_0_2022-12-12_20-32-45"
# model_dir = "ray_results_ppo/Jump2D-1216-12-27/PPO/PPO_DummyQuadrupedEnv_1c6d0_00000_0_2022-12-16_12-27-50" # 0.6 checkpoint 650
# model_dir = "ray_results_ppo/Jump2D-1215-23-04/PPO/PPO_DummyQuadrupedEnv_d3846_00000_0_2022-12-15_23-04-04" # 0.4
# model_dir = "ray_results_ppo/Jump2D-1215-17-30/PPO/PPO_DummyQuadrupedEnv_3b4f4_00000_0_2022-12-15_17-30-32" # 0.2
# model_dir = "ray_results_ppo/Jump2D-1216-17-05/PPO/PPO_DummyQuadrupedEnv_e7647_00000_0_2022-12-16_17-05-31" # 0.8
# model_dir = "ray_results_ppo/Jump2D-1216-23-12/PPO/PPO_DummyQuadrupedEnv_2baa3_00000_0_2022-12-16_23-12-30" # 1.0
# model_dir = "ray_results_ppo/Jump2D-1217-16-01/PPO/PPO_DummyQuadrupedEnv_32d69_00000_0_2022-12-17_16-02-01"
# model_dir = "ray_results_ppo/Jump2D-1217-18-36/PPO/PPO_DummyQuadrupedEnv_c18a7_00000_0_2022-12-17_18-36-20" # 0.4, train with pre land
# model_dir = "ray_results_ppo/Jump2D-1217-21-15/PPO/PPO_DummyQuadrupedEnv_0119b_00000_0_2022-12-17_21-15-36" # 0.8, train with pre land
# model_dir = "ray_results_ppo/Jump2D-1219-23-39/PPO/PPO_DummyQuadrupedEnv_6a2b2_00000_0_2022-12-19_23-39-20"
# model_dir = "ray_results_ppo/MPC-0116-17-18/PPO/PPO_DummyQuadrupedEnv_db429_00000_0_2023-01-16_17-18-32"
# model_dir = "ray_results_ppo/MPC-0118-15-02/PPO/PPO_DummyQuadrupedEnv_20236_00000_0_2023-01-18_15-02-05"
# model_dir = "ray_results_ppo/MPC-0118-21-39/PPO/PPO_DummyQuadrupedEnv_a981c_00000_0_2023-01-18_21-39-37"
# model_dir = "ray_results_ppo/MPC-Adaptive-MLP-0119-14-53/PPO/PPO_DummyQuadrupedEnv_245a4_00000_0_2023-01-19_14-53-51"
# model_dir = "ray_results_ppo/MPC-Adaptive-MLP-0126-22-10/PPO/PPO_DummyQuadrupedEnv_44356_00000_0_2023-01-26_22-10-15"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0421-22-12/PPO/PPO_DummyQuadrupedEnv_54ca7_00000_0_2023-04-21_22-12-52"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0422-17-39/PPO/PPO_DummyQuadrupedEnv_53a72_00000_0_2023-04-22_17-39-38"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0422-22-31/PPO/PPO_DummyQuadrupedEnv_23834_00000_0_2023-04-22_22-31-46"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0424-02-13/PPO/PPO_DummyQuadrupedEnv_38186_00000_0_2023-04-24_02-13-04"
# model_dir = "ray_results_ppo/MPC-accel-and-foot-0424-09-40/PPO/PPO_DummyQuadrupedEnv_c8ac4_00000_0_2023-04-24_09-40-55"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0425-00-20/PPO/PPO_DummyQuadrupedEnv_98d2c_00000_0_2023-04-25_00-20-03"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0426-00-18/PPO/PPO_DummyQuadrupedEnv_8a732_00000_0_2023-04-26_00-18-28"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0427-20-55/PPO/PPO_DummyQuadrupedEnv_93065_00000_0_2023-04-27_20-55-54"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0428-23-49/PPO/PPO_DummyQuadrupedEnv_077d3_00000_0_2023-04-28_23-49-46"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0430-00-20/PPO/PPO_DummyQuadrupedEnv_7c8d5_00000_0_2023-04-30_00-20-29"
# model_dir = "ray_results_ppo/MPC-accel-and-foot-0502-00-58/PPO/PPO_DummyQuadrupedEnv_2ec73_00000_0_2023-05-02_00-58-54"
# model_dir = "ray_results_ppo/MPC-accel-and-foot-0503-00-24/PPO/PPO_DummyQuadrupedEnv_7c264_00000_0_2023-05-03_00-24-05"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0503-20-59/PPO/PPO_DummyQuadrupedEnv_1145d_00000_0_2023-05-03_20-59-28"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0504-01-49/PPO/PPO_DummyQuadrupedEnv_8f72e_00000_0_2023-05-04_01-49-20"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0504-12-07/PPO/PPO_DummyQuadrupedEnv_ec861_00000_0_2023-05-04_12-07-33"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0504-15-35/PPO/PPO_DummyQuadrupedEnv_07efa_00000_0_2023-05-04_15-35-54"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0504-19-12/PPO/PPO_DummyQuadrupedEnv_3ad44_00000_0_2023-05-04_19-12-04"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0505-00-33/PPO/PPO_DummyQuadrupedEnv_24fa2_00000_0_2023-05-05_00-33-35"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0505-09-02/PPO/PPO_DummyQuadrupedEnv_4c90f_00000_0_2023-05-05_09-02-56"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0506-17-44/PPO/PPO_DummyQuadrupedEnv_4ae92_00000_0_2023-05-06_17-44-15"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0508-01-08/PPO/PPO_DummyQuadrupedEnv_83c4c_00000_0_2023-05-08_01-08-28"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0508-14-37/PPO/PPO_DummyQuadrupedEnv_87b92_00000_0_2023-05-08_14-37-28"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0508-17-37/PPO/PPO_DummyQuadrupedEnv_aee8a_00000_0_2023-05-08_17-37-31"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0509-00-03/PPO/PPO_DummyQuadrupedEnv_8c446_00000_0_2023-05-09_00-03-06"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0509-10-33/PPO/PPO_DummyQuadrupedEnv_8d7bb_00000_0_2023-05-09_10-33-04"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0509-00-03/PPO/PPO_DummyQuadrupedEnv_8c446_00000_0_2023-05-09_00-03-06"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0511-11-37/PPO/PPO_DummyQuadrupedEnv_dc78d_00000_0_2023-05-11_11-37-19"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0512-13-45/PPO/PPO_DummyQuadrupedEnv_ea0f8_00000_0_2023-05-12_13-45-22"
model_dir = "ray_results_ppo/MPC-accel-and-foot-0512-23-21/PPO/PPO_DummyQuadrupedEnv_57a0f_00000_0_2023-05-12_23-21-05"

model_dir = "ray_results_ppo/MPC-accel-and-foot-adaptive-0514-10-35/PPO/PPO_DummyQuadrupedEnv_b0512_00000_0_2023-05-14_10-35-16"
model_dir = "ray_results_ppo/MPC-accel-and-foot-adaptive-0515-11-39/PPO/PPO_DummyQuadrupedEnv_decf2_00000_0_2023-05-15_11-39-48"

model_dir = "ray_results_ppo/MPC-accel-and-foot-0516-10-47/PPO/PPO_DummyQuadrupedEnv_bf99a_00000_0_2023-05-16_10-47-38"
model_dir = "ray_results_ppo/MPC-accel-and-foot-adap-terrain-0522-02-38/PPO/PPO_DummyQuadrupedEnv_73f16_00000_0_2023-05-22_02-38-48"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0604-13-59/PPO/PPO_DummyQuadrupedEnv_ba87b_00000_0_2023-06-04_13-59-42"
# model_dir = "ray_results_ppo/MPC-accel-and-foot-adap-terrain-0522-02-38/PPO/PPO_DummyQuadrupedEnv_73f16_00000_0_2023-05-22_02-38-48"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0623-20-27/PPO/PPO_DummyQuadrupedEnv_0e2f1_00000_0_2023-06-23_20-27-32"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0626-18-09/PPO/PPO_DummyQuadrupedEnv_38f92_00000_0_2023-06-26_18-09-10"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0623-15-46/PPO/PPO_DummyQuadrupedEnv_daa49_00000_0_2023-06-23_15-46-55"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0628-22-48/PPO/PPO_DummyQuadrupedEnv_8d6a8_00000_0_2023-06-28_22-48-20"
model_dir = "ray_results_ppo/MPC-foot-0708-19-57/PPO/PPO_DummyQuadrupedEnv_59641_00000_0_2023-07-08_19-57-32"
model_dir = "ray_results_ppo/MPC-foot-react-0711-10-06/PPO/PPO_DummyQuadrupedEnv_3e9bf_00000_0_2023-07-11_10-06-15"
model_dir = "ray_results_ppo/MPC-60ms-adap-terrain-0709-17-33/PPO/PPO_DummyQuadrupedEnv_73cac_00000_0_2023-07-09_17-33-55"
model_dir = "ray_results_ppo/MPC-adaptive-0721-09-14/PPO/PPO_DummyQuadrupedEnv_aaf6e_00000_0_2023-07-21_09-14-28"
# model_dir = "ray_results_ppo/MPC-foot-react-0730-20-43/PPO/PPO_DummyQuadrupedEnv_798f9_00000_0_2023-07-30_20-43-55"
model_dir = "ray_results_ppo/MPC-foot-react-0731-00-47/PPO/PPO_DummyQuadrupedEnv_6fdc4_00000_0_2023-07-31_00-47-02"
model_dir = "ray_results_ppo/MPC-fast-running-new-0816-22-46/PPO/PPO_DummyQuadrupedEnv_7aa8f_00000_0_2023-08-16_22-46-58"
model_dir = "ray_results_ppo/MPC-fast-running-new-0817-10-47/PPO/PPO_DummyQuadrupedEnv_27667_00000_0_2023-08-17_10-47-37"
# model_dir = "ray_results_ppo/MPC-foot-react-0819-19-29/PPO/PPO_DummyQuadrupedEnv_66835_00000_0_2023-08-19_19-29-34"
model_dir = "ray_results_ppo/MPC-adaptive-0821-23-11/PPO/PPO_DummyQuadrupedEnv_b3d53_00000_0_2023-08-21_23-11-16"
# model_dir = "ray_results_ppo/MPC-foot-react-0824-00-02/PPO/PPO_DummyQuadrupedEnv_404f8_00000_0_2023-08-24_00-02-56"
# model_dir = "ray_results_ppo/MPC-foot-react-0825-13-47/PPO/PPO_DummyQuadrupedEnv_a0aaa_00000_0_2023-08-25_13-47-38"
# model_dir = "ray_results_ppo/MPC-fast-running-new-0818-00-51/PPO/PPO_DummyQuadrupedEnv_0d90a_00000_0_2023-08-18_00-51-34"
model_dir = "ray_results_ppo/MPC-fast-running-new-0926-16-11/PPO/PPO_DummyQuadrupedEnv_06dd1_00000_0_2023-09-26_16-11-30"
def run_sim(env, agent, count):
    rewards = []
    episode_lengths = []

    for i in range(count):
        print("Current episode: {}, remaining episodes: {}".format(i + 1, count - i - 1))
        obs = env.reset()
    
        foot_pos = []
        # obs_arr = [np.copy(obs)]
        episode_reward = 0
        num_steps = 0
        infer_time = 0
        actions = []
        while True:
            start_time = time.time()
            action = agent.compute_single_action(obs, explore=False)
            actions.append(action)
            infer_time += time.time() - start_time
            # action = agent.compute_action(obs)
            # action = np.array(env.action_space)
            obs, reward, done, info = env.step(action)
            
            # print('step reward ', reward)
            foot_pos.append(np.clip(action, env.action_space.low, env.action_space.high))
            # obs_arr.append(np.copy(obs))
            # time.sleep(0.01)
            # print(obs)
            episode_reward += reward
            num_steps += 1
            if num_steps % 100 == 0:
                a = 0
            #     env.env.save_robot_cam_view()
            if done:
                print('episode reward:', episode_reward, "num_steps:", num_steps)
                print("---Avg exec time per second: %.4f seconds ---" % (infer_time / num_steps))
                episode_lengths.append(num_steps)
                rewards.append(episode_reward)
                print(info)
                # plot_foot_pos(foot_pos)
                # if num_steps < 1001:
                #     time.sleep(5)
                # df = pd.DataFrame(np.array(actions))
                # df.to_csv("/home/zhuochen/actions" + str(i) + ".csv", index=False)
                break
        # df = pd.DataFrame(np.array(obs_arr))
        # df.to_csv("/home/zhuochen/obs_vision_" + str(i) + ".csv", index=False)
    return rewards, episode_lengths


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """

    def __init__(self, dummy_env_config):
        # self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=20, obs_hist_len=3)
        # self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=10, obs_hist_len=5) # joint space mlp
        self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=30, obs_hist_len=5)
        # self.env = QuadrupedManipEnv(render=True, time_step=0.001, action_repeat=50, obs_hist_len=10)
        # self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=20, obs_hist_len=5) #jumping
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # print('\n', '*' * 80)
        # print(self.observation_space)

    def reset(self):
        """reset env """
        obs = self.env.reset()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # print('step obs, rew, done, info', obs, rew, done, info)
        # NOTE: it seems pybullet torque control IGNORES joint velocity limits..
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if (obs < self.observation_space.low).any() or (obs > self.observation_space.high).any():
            print(obs)
            sys.exit()
        return np.array(obs), rew, done, info

register_env("quadruped_env", lambda _: DummyQuadrupedEnv(_))

ModelCatalog.register_custom_model("my_model", MyFullyConnectedNetwork)
# ModelCatalog.register_custom_model("my_model", LSTMModel)

config_path = os.path.join(model_dir, "params.pkl")

# config_path = "/home/zhuochen/params.pkl"

with open(config_path, "rb") as f:
    config = pickle.load(f)
config["num_workers"] = 0
config["num_gpus"] = 0
# config["model"] = {"custom_model": "my_model", "fcnet_activation": "tanh", "vf_share_layers": True}
#                    "custom_model_config": {"cam_seq_len": 10, "sensor_seq_len": 30, "action_seq_len": 1,
#                                            "cam_dim": (36, 36),
                                        #    "is_training": False}}

config["num_envs_per_worker"] = 1
config["evaluation_config"] = {
    "explore": False,
    "env_config": {
        # Use test set to evaluate
        'mode': "test"}
}

ray.init()
agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)

latest = 'checkpoint_003000/checkpoint-3000'
# checkpoint = get_latest_model_rllib(model_dir)
checkpoint = os.path.join(model_dir, latest)
# checkpoint = "/home/zhuochen/checkpoint_120/checkpoint-120"
agent.restore(checkpoint)
env = agent.workers.local_worker().env
rllib_mean_std_filter = agent.workers.local_worker().filters['default_policy']
variance = list(np.square(rllib_mean_std_filter.rs.std))
mean = list(rllib_mean_std_filter.rs.mean)

policy_weights = agent.get_policy().get_weights()
policy_dict = dict()
policy_dict["obs_mean"] = mean
policy_dict["obs_variance"] = variance
for i in range(len(policy_weights) - 2):
    policy_dict[("layer" + str(i) + "/shape")] = list(policy_weights[i].shape)
    policy_dict[("layer" + str(i) + "/value")] = policy_weights[i].tolist()

print(policy_dict)
# with open("foot_reaction_test.json", "w") as outfile:
with open("run_and_turn.json", "w") as outfile:
    json.dump(policy_dict, outfile)
outfile.close()

# print(action)
start_time = time.time()
rewards, lengths = run_sim(env, agent, 1)
end_time = time.time()
# print("total_time", end_time - start_time)
# generate(env, 5000)
# print("Average reward: {}, average episode length: {}".format(np.mean(rewards), np.mean(lengths)))

# ray.shutdown()
