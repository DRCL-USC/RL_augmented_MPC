"""rllib utils """
import os
import gym
import numpy as np
# quadruped env
from usc_learning.envs.quadruped_master.quadruped_gym_env import QuadrupedGymEnv
#from usc_learning.ros_interface.ros_quadruped_env import ROSQuadrupedGymEnv
from usc_learning.imitation_tasks.imitation_gym_env import ImitationGymEnv

# stable baselines vec env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from ray.tune import Stopper
from collections import deque

from usc_learning.utils.file_utils import get_latest_model_rllib, get_latest_directory, write_env_config
#from usc_learning.utils.file_utils import copy_files, write_env_config

USING_VEC_ENV = True


##########################################################################################################
# Dummy env interface for rllib
##########################################################################################################

class DummyQuadrupedEnv(gym.Env):
    """ Dummy class to work with rllib. """
    def __init__(self, env_config):
        #env = QuadrupedGymEnv(render=False,hard_reset=True)
        if USING_VEC_ENV:
            # same as in stable baselines
            # print('*\n'*10)
            # print(env_config)
            if env_config['USE_ROS']:
                env = lambda: ROSQuadrupedGymEnv(**env_config)
            elif env_config['USE_IMITATION_ENV']:
                env = lambda: ImitationGymEnv(**env_config)
            else:
                env = lambda: QuadrupedGymEnv(**env_config)
            env = make_vec_env(env, monitor_dir=env_config['vec_save_path'])

            try: # try to load stats
                env = VecNormalize.load(env_config['vec_load_path'], env)
                env.training = True
                print('LOADED STATS')
                #sys.exit()
            except:
                env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=1, clip_obs=100.)

            self.env = env 
            self._save_vec_stats_counter = 0
            self.vec_stats_file = os.path.join(env_config['vec_save_path'], "vec_normalize.pkl")
            # write env configs to reload (i.e if don't need to inspect whole code base)
            write_env_config(env_config['vec_save_path'],env)
        else:
            raise ValueError('Env should be vec normalized.')
            self.env = QuadrupedGymEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # print('\n','*'*80)
        # print('obs high',self.observation_space.high)
        # print('obs low', self.observation_space.low)

    def reset(self):
        """reset env """
        obs = self.env.reset()

        if USING_VEC_ENV:
            obs = obs[0]
            self._save_vec_stats_counter += 1
            #print('reset env', self._save_vec_stats_counter)
            #save vec normalize stats every so often
            if self._save_vec_stats_counter > 10:
                #print('saving to ', self.vec_stats_file)
                self.env.save(self.vec_stats_file)
                self._save_vec_stats_counter = 0

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        """step env """
        if USING_VEC_ENV:
            # note the VecEnv list convention
            obs, rew, done, info = self.env.step([action])
            obs = obs[0]
            rew = rew[0]
            done = done[0]
            info = info[0]
            #print(obs, rew, done, info )
        else:
            obs, rew, done, info = self.env.step(action)
        # just as a sanity check
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        if any(obs < self.observation_space.low) or any(obs > self.observation_space.high):
            print(obs)
            sys.exit()
        return obs, rew, done, info 


##########################################################################################################
# Custom stopper for training rllib with tune
##########################################################################################################

class CustomStopper(Stopper):
    """Custom stopper, let's stop after some pre-determined number of trials, OR if the mean reward is not moving """
    def __init__(self, iteration_timesteps=1):
        self.should_stop = False
        self.rew_means = deque(maxlen=20)
        self.ep_lens = deque(maxlen=20)
        self.mean_diff = 5 # check if stagnant - not learning anything
        self._max_timesteps_per_space = iteration_timesteps * 10000
        self._prev_trial_avg = []
        self._prev_trial_len_avg = []
        self._total_timesteps = 0

        self._MAX_GAS = 200000

    def __call__(self, trial_id, result):
        """We should only return once the new action space is doing at least as well as the old one. """
        self.should_stop = False
        self.rew_means.extend([result["episode_reward_mean"]])
        self.ep_lens.extend([result["episode_len_mean"]])
        self._total_timesteps = result['timesteps_total']
        prev_mean = self.safemean(self.rew_means)
        curr_mean = result["episode_reward_mean"]
        
        #if abs(curr_mean-prev_mean) < self.mean_diff or result['timesteps_total'] > self._max_timesteps_per_space:
        if len(self._prev_trial_avg) != 0:
            # check how we are doing compared to previous trial, and make sure we have done at least number of trials
            # if (abs(curr_mean-self._prev_trial_avg[0]) < self.mean_diff or curr_mean > self._prev_trial_avg[0]) \
            #     and result['timesteps_total'] > self._max_timesteps_per_space \
            #     and result["episode_len_mean"] >= self._prev_trial_len_avg[0]:
            #     self.should_stop = True
            if ((curr_mean > np.mean(self._prev_trial_avg) - self.mean_diff) \
                or result["episode_len_mean"] >= np.mean(self._prev_trial_len_avg) - 5) \
                and result['timesteps_total'] > self._max_timesteps_per_space:
                self.should_stop = True
                print('******************************\ngrowing action space')
                print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
                print('******************************')

            # just to test
            # if result['timesteps_total'] > self._max_timesteps_per_space:
            #     self.should_stop = True
            #     print('******************************\ngrowing action space')
            #     print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
            #     print('******************************')

            # if self._total_timesteps > self._MAX_GAS:
            #     self.should_stop = True
            #     print('******************************\nworse performance in this new action range - growing anyways')
            #     print('been at this action space way too long, shrink back down? or increase anyways')
            #     print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
            #     print('******************************')
        elif result['timesteps_total'] > self._max_timesteps_per_space:
            self.should_stop = True
            print('******************************\ngrowing action space')
            print('prev_mean', prev_mean, 'curr_mean',curr_mean, 'total timesteps', result['timesteps_total'])
            print('******************************')

        print('**'*50,'\n',' -'*50)
        print('current stats to beat:')
        print( 'prev best', self._prev_trial_avg, 'current mean',curr_mean)
        print('last 20:', self.rew_means )
        print('LENGTHS', self._prev_trial_len_avg, 'current length', result["episode_len_mean"] )
        print('total timesteps', self._total_timesteps, 'max GAS timesteps', self._MAX_GAS)
        print(' -'*50,'\n','**'*50)
        # if not self.should_stop and result['foo'] > 10:
        #     self.should_stop = True
        return self.should_stop

    def set_additional_time_steps(self,num_timesteps):
        self.should_stop = False
        self._prev_trial_len_avg.append(max(self.ep_lens[-1], self.safemean(self.ep_lens)))
        self._prev_trial_avg.append(max(self.rew_means[-1], self.safemean(self.rew_means))) # self.safemean(self.rew_means)
        #self._max_timesteps_per_space = num_timesteps
        self._max_timesteps_per_space = self._total_timesteps + num_timesteps
        self._MAX_GAS = self._total_timesteps + 200000

    def safemean(self,xs):
        """Avoid division by zero """
        return np.nan if len(xs) == 0 else np.mean(xs)

    def stop_all(self):
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop


##########################################################################################################
# [TODO: TEST] export rllib model 
##########################################################################################################

def export_rllib_tf(agent, env, write_out_path='./exported_model/'):
    """Write out tf graph, variables, vecnorm params and observation and action spaces. 
    For loading in ROS

    Currently for IMPEDANCE action space only, will need a flag for this..
    """

    from ray.tune.trial import ExportFormat
    import tensorflow as tf
    #agent.export_policy_model('./temp69')
    #agent.export_model([ExportFormat.CHECKPOINT, ExportFormat.MODEL],'./temp5')
    policy = agent.get_policy()
    # print(policy)
    # print(policy.get_session())
    # print('variables', policy.variables())
    # #agent.export_policy_checkpoint('./temp11')
    # #with policy.get_session():
    # i = tf.initializers.global_variables()
    # with open('./temp11/model.pb', 'wb') as f:
    #     f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    #saver = tf.train.Saver(tf.all_variables())
    with policy.get_session() as sess:
        saver = tf.train.Saver(policy.variables())#tf.global_variables())
        # saver.save(sess, './exported/my_model')
        # tf.train.write_graph(sess.graph, '.', './exported/graph.pb', as_text=False)
        saver.save(sess, write_out_path + 'my_model')
        tf.train.write_graph(sess.graph, '.', write_out_path + 'graph.pb', as_text=False)

    # save observation and action space 
    venv = env.env.env.env
    quadruped_env = env.env.env.env.venv.envs[0].env
    print('mean', venv.obs_rms.mean)
    print('var ', venv.obs_rms.var)
    vecnorm_arr = np.vstack((venv.obs_rms.mean, 
                            venv.obs_rms.var,
                            venv.observation_space.high,
                            venv.observation_space.low))
    act_space_arr = np.vstack((
                            np.concatenate((quadruped_env._robot_config.IK_POS_UPP_LEG,quadruped_env._robot_config.MAX_NORMAL_FORCE)),
                            np.concatenate((quadruped_env._robot_config.IK_POS_LOW_LEG,quadruped_env._robot_config.MIN_NORMAL_FORCE))
                            ))
    # np.savetxt('./exported/vecnorm_params.csv',vecnorm_arr,delimiter=',')
    # np.savetxt('./exported/action_space.csv', act_space_arr, delimiter=',')
    np.savetxt(write_out_path + 'vecnorm_params.csv',vecnorm_arr,delimiter=',')
    np.savetxt(write_out_path + 'action_space.csv', act_space_arr, delimiter=',')

##########################################################################################################
# helper to get checkpoints 
##########################################################################################################
def get_checkpoint_from_last_GAS(trial_num,save_results_path=None,alg=None):
    """Return latest check point  """
    if trial_num < 0: # first trial
        return None
    # now we have a previous trial
    train_dir = get_latest_directory( os.path.join(save_results_path,str(trial_num),alg) )
    print('train_dir', train_dir)
    # get checkpoint
    checkpoint = get_latest_model_rllib(train_dir)
    return checkpoint
