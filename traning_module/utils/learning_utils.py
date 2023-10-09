"""Training utils. """
import os, sys
from stable_baselines.common.callbacks import BaseCallback
import numpy as np

##########################################################################################################
# Checkpoint callback for stable baselines
##########################################################################################################

class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model and vec_env parameters every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):# -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):# -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)

            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.training_env.save(stats_path)

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))

        if self.save_freq < 10000 and self.n_calls % 50 == 0:
            # also print out path periodically for off-policy aglorithms: SAC, TD3, etc.
            print('=================================== Save path is {}'.format(self.save_path))
        return True




class GASCallback(CheckpointCallback):
    """
    Callback for saving a model and vec_env parameters every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix='rl_model', verbose=0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):# -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):# -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, self.num_timesteps))
            self.model.save(path)

            stats_path = os.path.join(self.save_path, "vec_normalize.pkl")
            self.training_env.save(stats_path)

            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

##########################################################################################################
# Stable baselines alg configs
##########################################################################################################

def get_ppo2_configs():
    # PPO2 (Default)
    # ppo_config = {"gamma":0.99, "n_steps":512, "ent_coef":0.01, "learning_rate":2.5e-3, "vf_coef":0.5,
    #               "max_grad_norm":0.5, "lam":0.95, "nminibatches":32, "noptepochs":4, "cliprange":0.2, "cliprange_vf":None,
    #               "verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":policy_kwargs,
    #               "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}
    # lambda f: 2e-3 * f
    # these settings seem to be working well, with NUM_CPU=4
    n_steps = 1024
    nminibatches = int(n_steps / 64)
    learning_rate = lambda f: 1e-4 #5e-4 * f
    ppo_config = {"gamma":0.99, "n_steps":n_steps, "ent_coef":0.0, "learning_rate":learning_rate, "vf_coef":0.5,
                    "max_grad_norm":0.5, "lam":0.95, "nminibatches":nminibatches, "noptepochs":10, "cliprange":0.2, "cliprange_vf":None,
                    "verbose":1, "tensorboard_log":"logs/ppo2/tensorboard/", "_init_setup_model":True, "policy_kwargs":{},
                    "full_tensorboard_log":False, "seed":None, "n_cpu_tf_sess":None}


##########################################################################################################
# Load stable baselines model
##########################################################################################################
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO1,PPO2, SAC
from stable_baselines.common.cmd_util import make_vec_env
from usc_learning.utils.utils import plot_results
from usc_learning.utils.file_utils import get_latest_model, read_env_config
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv

def get_stable_baselines_model_env(log_dir, LEARNING_ALG=0):
    """Get model and env, stable baselines version. 
    LEARNING_ALG 
    0 PPO2
    1 PPO1
    2 = SAC
    """

    # get env config if available
    try:
        env_config = read_env_config(log_dir)
        sys.path.append(log_dir)
        print('env_config path', env_config)

        # check env_configs.txt file to see whether to load aliengo or a1
        with open(log_dir + '/env_configs.txt') as configs_txt:
            if 'aliengo' in configs_txt.read():
                import configs_aliengo as robot_config
            else:
                import configs_a1 as robot_config

        env_config['robot_config'] = robot_config
    except:
        env_config = {}
    #env_config['render'] = True
    env_config['record_video'] = False
    for k,v in env_config.items():
        print(k,v)

    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)
    #monitor_results = load_results(log_dir)

    # reconstruct env 
    env = lambda: QuadrupedGymEnv(**env_config)
    env = make_vec_env(env, n_envs=1)
    env = VecNormalize.load(stats_path, env)
    # do not update stats at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    if LEARNING_ALG == 0:
        model = PPO2.load(model_name, env)
    elif LEARNING_ALG == 1:
        model = PPO1.load(model_name, env)
    elif LEARNING_ALG == 2:
        model = SAC.load(model_name, env)

    return model, env


##########################################################################################################
# Load rllib  model
##########################################################################################################
import pickle
import gym
from collections.abc import Mapping
import ray
from ray.rllib.agents import ppo 
from usc_learning.utils.file_utils import get_latest_directory, get_latest_model_rllib

USING_VEC_ENV = True

def get_rllib_model_env(log_dir, USING_VEC_ENV=True, LEARNING_ALG=0):
    alg_dir = get_latest_directory(log_dir) #i.e. {log_dir}/PPO
    train_dir = get_latest_directory(alg_dir) # i.e. {log_dir}/PPO/PPO_DummyQuadrupedEnv_0_2020-07-11_13-27-26rtt2kwpd

    # most recent checkpoint in train_dir
    checkpoint = get_latest_model_rllib(train_dir)
    # visualize episode lengths/rewards over training
    #data = load_rllib(train_dir)

    # for loading env configs, vec normalize stats
    stats_path = str(log_dir)
    vec_stats_path = os.path.join(stats_path,"vec_normalize.pkl")

    # get env config if available
    try:
        # stats path has all saved files (quadruped, configs_a1, as well as vec params)
        print('stats_path', stats_path)
        env_config = read_env_config(stats_path)
        print('env config', env_config)
        sys.path.append(stats_path)

        # check env_configs.txt file to see whether to load aliengo or a1
        with open(stats_path + '/env_configs.txt') as configs_txt:
            if 'aliengo' in configs_txt.read():
                import configs_aliengo as robot_config
            else:
                import configs_a1 as robot_config

        env_config['robot_config'] = robot_config
    except:
        print('*'*50,'could not load env stats','*'*50)
        env_config = {}
        sys.exit()

    # toggle if using with ROS
    #env_config['render'] = True

    class DummyQuadrupedEnv(gym.Env):
        """ Dummy class to work with rllib. """
        def __init__(self, dummy_env_config):#,vec_stats_path=None):
            # set up like stable baselines (will this work?)
            #env = QuadrupedGymEnv(render=False,hard_reset=True)
            if USING_VEC_ENV:
                env = lambda: QuadrupedGymEnv(**env_config) #render=True,hard_reset=True
                env = make_vec_env(env, n_envs=1)
                env = VecNormalize.load(vec_stats_path, env)
                # do not update stats at test time
                env.training = False
                # reward normalization is not needed at test time
                env.norm_reward = False

                self.env = env 
                #self._save_vec_stats_counter = 0
            else:
                self.env = QuadrupedGymEnv(**env_config)#render=True,hard_reset=True)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            print('\n','*'*80)
            print('obs high',self.observation_space.high)
            print('obs low', self.observation_space.low)

        def reset(self):
            """reset env """
            obs = self.env.reset()
            if USING_VEC_ENV:
                obs = obs[0]
                # no need to save anything at load time
            return np.clip(obs, self.observation_space.low, self.observation_space.high)

        def step(self, action):
            """step env """
            if USING_VEC_ENV:
                obs, rew, done, info = self.env.step([action])
                obs, rew, done, info = obs[0], rew[0], done[0], info[0]
            else:
                obs, rew, done, info = self.env.step(action)
            #print('step obs, rew, done, info', obs, rew, done, info)
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
            if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
                print(obs)
                sys.exit()
            return obs, rew, done, info 

    # load training configurations (can be seen in params.json)
    config_path = os.path.join(train_dir, "params.pkl")

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    # complains if 1.. but is should be 1?
    config["num_workers"] = 0

    #ray.init()
    #agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv(env_config,vec_stats_path=vec_stats_path))
    agent = ppo.PPOTrainer(config=config, env=DummyQuadrupedEnv)
    # Load state from checkpoint.
    agent.restore(checkpoint)
    # env to pass to ROS interface class
    rl_env = agent.workers.local_worker().env

    return agent, rl_env



# class DummyQuadrupedEnv(gym.Env):
#     """ Dummy class to work with rllib. """
#     def __init__(self, env_config,vec_stats_path=None):
#         # set up like stable baselines (will this work?)
#         #env = QuadrupedGymEnv(render=False,hard_reset=True)
#         if USING_VEC_ENV:
#             env = lambda: QuadrupedGymEnv(**env_config) #render=True,hard_reset=True
#             env = make_vec_env(env, n_envs=1)
#             env = VecNormalize.load(vec_stats_path, env)
#             # do not update stats at test time
#             env.training = False
#             # reward normalization is not needed at test time
#             env.norm_reward = False

#             self.env = env 
#             #self._save_vec_stats_counter = 0
#         else:
#             self.env = QuadrupedGymEnv(**env_config)#render=True,hard_reset=True)
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space
#         print('\n','*'*80)
#         print('obs high',self.observation_space.high)
#         print('obs low', self.observation_space.low)

#     def reset(self):
#         """reset env """
#         obs = self.env.reset()
#         if USING_VEC_ENV:
#             obs = obs[0]
#             # no need to save anything at load time
#         return np.clip(obs, self.observation_space.low, self.observation_space.high)

#     def step(self, action):
#         """step env """
#         if USING_VEC_ENV:
#             obs, rew, done, info = self.env.step([action])
#             obs, rew, done, info = obs[0], rew[0], done[0], info[0]
#         else:
#             obs, rew, done, info = self.env.step(action)
#         #print('step obs, rew, done, info', obs, rew, done, info)
#         obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
#         if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
#             print(obs)
#             sys.exit()
#         return obs, rew, done, info 