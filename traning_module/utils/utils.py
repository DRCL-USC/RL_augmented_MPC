import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import pandas
import json

from stable_baselines.bench.monitor import load_results
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

""" utils.py - general utilities """

################################################################
## Printing
################################################################

def nicePrint(vec):
    """ Print single vector (list, tuple, or numpy array) """
    # check if vec is a numpy array
    if isinstance(vec,np.ndarray):
        np.set_printoptions(precision=3)
        print(vec)
        return
    currStr = ''
    for x in vec:
        currStr = currStr + '{: .3f} '.format(x)
    print(currStr)

def nicePrint2D(vec):
    """ Print 2D vector (list of lists, tuple of tuples, or 2D numpy array) """
    for x in vec:
        currStr = ''
        for y in x:
            currStr = currStr + '{: .3f} '.format(y)
        print(currStr)


################################################################
## Plotting
################################################################
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
Y_EPLEN = True
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray

    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays

    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


def ts2xy(timesteps, xaxis,yaxis=None):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    if yaxis is Y_EPLEN:
        y_var = timesteps.l.values
    return x_var, y_var


def plot_curves(xy_list, xaxis, title):
    """
    plot the curves

    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()

def plot_results(dirs, num_timesteps, xaxis, task_name):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    #plt.figure(1)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Rewards')
    plt.ylabel("Episode Rewards")
    #plt.figure(2)
    xy_list = [ts2xy(timesteps_item, xaxis, Y_EPLEN) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name+'Ep Len')
    plt.ylabel("Episode Length")


######################################################################################
## Load progress/result files (make general so can use from stable-baselines or rllib)
######################################################################################
# from typing import Tuple, Dict, Any, List, Optional
# EXT = "monitor.csv"

# def get_monitor_files(path: str) -> List[str]:
#     """
#     get all the monitor files in the given path

#     :param path: (str) the logging folder
#     :return: ([str]) the log files
#     """
#     return glob(os.path.join(path, "*" + EXT))

# def load_rl_results(path: str) -> pandas.DataFrame:
#     """
#     Load all Monitor logs from a given directory path matching ``*monitor.csv`` and ``*monitor.json``

#     :param path: (str) the directory path containing the log file(s)
#     :return: (pandas.DataFrame) the logged data
#     """
#     # get both csv and (old) json files
#     monitor_files = (glob(os.path.join(path, "*monitor.json")) + get_monitor_files(path))
#     if not monitor_files:
#         raise ValueError("no monitor files of the form *%s found in %s" % (EXT, path))
#     data_frames = []
#     headers = []
#     for file_name in monitor_files:
#         with open(file_name, 'rt') as file_handler:
#             if file_name.endswith('csv'):
#                 first_line = file_handler.readline()
#                 assert first_line[0] == '#'
#                 header = json.loads(first_line[1:])
#                 data_frame = pandas.read_csv(file_handler, index_col=None)
#                 headers.append(header)
#             elif file_name.endswith('json'):  # Deprecated json format
#                 episodes = []
#                 lines = file_handler.readlines()
#                 header = json.loads(lines[0])
#                 headers.append(header)
#                 for line in lines[1:]:
#                     episode = json.loads(line)
#                     episodes.append(episode)
#                 data_frame = pandas.DataFrame(episodes)
#             else:
#                 assert 0, 'unreachable'
#             data_frame['t'] += header['t_start']
#         data_frames.append(data_frame)
#     data_frame = pandas.concat(data_frames)
#     data_frame.sort_values('t', inplace=True)
#     data_frame.reset_index(inplace=True)
#     data_frame['t'] -= min(header['t_start'] for header in headers)
#     # data_frame.headers = headers  # HACK to preserve backwards compatibility
#     return data_frame


def load_rllib(path: str) -> pandas.DataFrame:
    """
    Load progress.csv and result.json file

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    progress_file = os.path.join(path, "progress.csv")
    result_file = os.path.join(path, "result.json")

    data_frames = []
    headers = []

    with open(progress_file, 'rt') as file_handler:
        data_frame = pandas.read_csv(file_handler, index_col=None)
        #headers.append(header)
        # for col in data_frame.columns:
        #   print(col)
        #   plt.plot(data_frame['timesteps_total'], data_frame[col])
        #   plt.title(col)
        #   plt.show()

        plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_mean'], label='episode_reward_mean')
        try:
            plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_max'], label='episode_reward_max')
            plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_min'], label='episode_reward_min')
        except:
            pass
        plt.legend()
        plt.title('Episode reward stats')
        plt.show()

        plt.plot(data_frame['timesteps_total'], data_frame['episode_len_mean'], label='episode_len_mean')
        plt.legend()
        plt.title('Episode length')
        plt.show()

    try: 
        with open(result_file, 'rt') as file_handler: 
            # result.json, check it out
            all_episode_lengths = []
            all_episode_rewards = []
            timestep_totals = []
            # read in data
            line = file_handler.readline()
            while line: 
                ep_data = json.loads(line)
                # for k,v in ep_data.items():
                #     print(k,v)
                # print(ep_data)

                eplens = ep_data['hist_stats']['episode_lengths']
                eprews = ep_data['hist_stats']['episode_reward']
                # at the beginning will have simulated more than 100 episodes due to early terminations
                episodes_this_iter = min(ep_data['episodes_this_iter'],len(eplens))
                # buffer has previous 100 episodes, which have mostly already been counted, so just display new ones
                eplens = eplens[:episodes_this_iter]
                eprews = eprews[:episodes_this_iter]

                all_episode_lengths.extend(eplens)
                all_episode_rewards.extend(eprews)
                timestep_totals.extend( [ep_data['timesteps_total']]*len(eplens))
                line = file_handler.readline()

            plt.scatter(timestep_totals, all_episode_rewards, s=2)
            x, y_mean = window_func(np.array(timestep_totals), 
                                    np.array(all_episode_rewards), 
                                    EPISODES_WINDOW, 
                                    np.mean)
            plt.plot(x, y_mean, color='red')
            plt.title('Episode Rewards')
            # plt.xlabel('Timesteps')
            # plt.ylabel('Avg. Reward',color='red')
            # plt.tick_params(axis='y', colors='red')
            plt.show()
            plt.scatter(timestep_totals, all_episode_lengths, s=2)
            x, y_mean = window_func(np.array(timestep_totals), 
                                    np.array(all_episode_lengths), 
                                    EPISODES_WINDOW, 
                                    np.mean)
            plt.plot(x, y_mean, color='red')
            plt.title('All Episode Lengths')
            plt.show()


            # episodes = []
            # lines = file_handler.readlines()
            # header = json.loads(lines[0])
            # headers.append(header)
            # for line in lines[1:]:
            #     episode = json.loads(line)
            #     episodes.append(episode)
            # data_frame = pandas.DataFrame(episodes)

            #data_frame['t'] += header['t_start']
            data_frames.append(data_frame)
        data_frame = pandas.concat(data_frames)
        #data_frame.sort_values('t', inplace=True)
        #data_frame.reset_index(inplace=True)
        #data_frame['t'] -= min(header['t_start'] for header in headers)
        # data_frame.headers = headers  # HACK to preserve backwards compatibility
        return data_frame

    except:
        print('WARNING: ES - so different data for loading result.json')
        return None


def load_rllib_v2(path: str) -> pandas.DataFrame:
    """
    Load progress.csv and result.json file, for 1 of several 

    :param path: (str) the directory path containing the log file(s)
    :return: (pandas.DataFrame) the logged data
    """
    # get both csv and (old) json files
    progress_file = os.path.join(path, "progress.csv")
    result_file = os.path.join(path, "result.json")

    data_frames = []
    headers = []

    with open(progress_file, 'rt') as file_handler:
        data_frame = pandas.read_csv(file_handler, index_col=None)
        #headers.append(header)
        # for col in data_frame.columns:
        #   print(col)
        #   plt.plot(data_frame['timesteps_total'], data_frame[col])
        #   plt.title(col)
        #   plt.show()

        # plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_mean'], label='episode_reward_mean')
        # plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_max'], label='episode_reward_max')
        # plt.plot(data_frame['timesteps_total'], data_frame['episode_reward_min'], label='episode_reward_min')
        # plt.legend()
        # plt.title('Episode reward stats')
        # plt.show()

        # plt.plot(data_frame['timesteps_total'], data_frame['episode_len_mean'], label='episode_len_mean')
        # plt.legend()
        # plt.title('Episode length')
        # plt.show()

    with open(result_file, 'rt') as file_handler: 
        # result.json, check it out
        all_episode_lengths = []
        all_episode_rewards = []
        timestep_totals = []
        # read in data
        line = file_handler.readline()
        while line: 
            ep_data = json.loads(line)
            # for k,v in ep_data.items():
            #     print(k,v)
            # print(ep_data)

            eplens = ep_data['hist_stats']['episode_lengths']
            eprews = ep_data['hist_stats']['episode_reward']
            # at the beginning will have simulated more than 100 episodes due to early terminations
            episodes_this_iter = min(ep_data['episodes_this_iter'],len(eplens))
            # buffer has previous 100 episodes, which have mostly already been counted, so just display new ones
            eplens = eplens[:episodes_this_iter]
            eprews = eprews[:episodes_this_iter]

            all_episode_lengths.extend(eplens)
            all_episode_rewards.extend(eprews)
            timestep_totals.extend( [ep_data['timesteps_total']]*len(eplens))
            line = file_handler.readline()

        # plt.scatter(timestep_totals, all_episode_rewards, s=2)
        # x, y_mean = window_func(np.array(timestep_totals), 
        #                         np.array(all_episode_rewards), 
        #                         EPISODES_WINDOW, 
        #                         np.mean)
        # plt.plot(x, y_mean, color='red')
        # plt.title('All Episode Rewards')
        # plt.show()
        # plt.scatter(timestep_totals, all_episode_lengths, s=2)
        # x, y_mean = window_func(np.array(timestep_totals), 
        #                         np.array(all_episode_lengths), 
        #                         EPISODES_WINDOW, 
        #                         np.mean)
        # plt.plot(x, y_mean, color='red')
        # plt.title('All Episode Lengths')
        # plt.show()


        # episodes = []
        # lines = file_handler.readlines()
        # header = json.loads(lines[0])
        # headers.append(header)
        # for line in lines[1:]:
        #     episode = json.loads(line)
        #     episodes.append(episode)
        # data_frame = pandas.DataFrame(episodes)

        #data_frame['t'] += header['t_start']
        data_frames.append(data_frame)
    data_frame = pandas.concat(data_frames)
    #data_frame.sort_values('t', inplace=True)
    #data_frame.reset_index(inplace=True)
    #data_frame['t'] -= min(header['t_start'] for header in headers)
    # data_frame.headers = headers  # HACK to preserve backwards compatibility
    return data_frame, timestep_totals, all_episode_rewards, all_episode_lengths


# def plot_GAS(log_dir, n):
#     """ Take in log directory and  """