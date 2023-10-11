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

# from envs.MPC_task import QuadrupedGymEnv
# from envs.MPC_foot_reaction_env import  QuadrupedGymEnv
from envs.MPC_task_accel_foothold import QuadrupedGymEnv

import pandas as pd
import torch
import matplotlib.pyplot as plt

from ray.rllib.models import ModelCatalog

from learning.rllib_helpers.fcnet_me import MyFullyConnectedNetwork

from matplotlib import pyplot as plt

# model_dir = "CHANGE TO CHECKPOINT DIRECTORY" #
model_dir = " "

latest = 'checkpoint_003000/checkpoint-3000' #set the check point here

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
            obs, reward, done, info = env.step(action)
            
            # print('step reward ', reward)
            foot_pos.append(np.clip(action, env.action_space.low, env.action_space.high))

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

                break
    return rewards, episode_lengths


class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """

    def __init__(self, dummy_env_config):
        self.env = QuadrupedGymEnv(render=True, time_step=0.001, action_repeat=30, obs_hist_len=5)
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
config["num_envs_per_worker"] = 1
config["evaluation_config"] = {
    "explore": False,
    "env_config": {
        # Use test set to evaluate
        'mode': "test"}
}

ray.init()
agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)


checkpoint = os.path.join(model_dir, latest)

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

with open("policy.json", "w") as outfile: #change the name of json file
    json.dump(policy_dict, outfile)
outfile.close()

start_time = time.time()
rewards, lengths = run_sim(env, agent, 1)
end_time = time.time()

