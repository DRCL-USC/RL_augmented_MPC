""" file management, copying etc. """
import os
import inspect 
import json
from shutil import copyfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Going to copy all relevant files
# envs/quadruped_master, imitation_tasks (if necessary)
import usc_learning.envs.quadruped_master as quadruped_master
import usc_learning.imitation_tasks as imitation_tasks
import usc_learning.learning as learning
import usc_learning.learning.rllib as learning_rllib

from stable_baselines3.common.monitor import load_results
from usc_learning.utils.utils import plot_curves, ts2xy

def copy_files(destination_directory):
	""" Copy relevant files to the log directory.
	Include envs/quadruped_master, imitation_tasks, run_rl, load_rl, rllib
	Can probably copy whole directory structure in future
	"""
	# get quadruped_master path
	envs_base_path = os.path.dirname(inspect.getfile(quadruped_master))
	files_to_save = [os.path.join(envs_base_path, f) for f in os.listdir(envs_base_path) if f.endswith('.py')]
	# also save imitation folder
	imitation_base_path = os.path.dirname(inspect.getfile(imitation_tasks))
	files_to_save.extend([os.path.join(imitation_base_path, f) for f in os.listdir(imitation_base_path) if f.endswith('.py')])
	# get learning files run_rl.py, load_rl.py
	learning_dir_path = os.path.dirname(inspect.getfile(learning))
	#files_to_save.append(os.path.join(learning_dir_path,'run_rl.py'))
	#files_to_save.append(os.path.join(learning_dir_path,'load_rl.py'))
	files_to_save.extend([os.path.join(learning_dir_path, f) for f in os.listdir(learning_dir_path) if f.endswith('.py')])
	# rllib too
	learning_rllib_dir_path = os.path.dirname(inspect.getfile(learning_rllib))
	files_to_save.append(os.path.join(learning_rllib_dir_path,'run_rllib.py'))
	files_to_save.append(os.path.join(learning_rllib_dir_path,'run_gas_clean.py'))

	# copy the files to log directory
	for f in files_to_save:
		filename = os.path.basename(f)
		#copyfile(f, os.path.join(os.path.dirname(os.path.abspath(__file__)),destination_directory,filename))
		print('copying',f, 'to', os.path.join(destination_directory,filename))
		copyfile(f, os.path.join(destination_directory,filename))


def write_env_config(destination_directory, vec_env, updated_config=None):
	"""Write configurations to file. """
	try:
		signature = inspect.getargspec(vec_env.venv.envs[0].env.__init__)
	except:
		# not using vec env
		signature = inspect.getargspec(vec_env.__init__)
	#print('signature', signature)
	#print('config', os.path.join(destination_directory,"env_configs.txt"))
	with open(os.path.join(destination_directory,"env_configs.txt"),"w") as outfile:
		args_dict = {}
		# skip urdf root and config file (should be obvious from context) for json
		for i,k,v in zip(range(len(signature.defaults)),signature.args[1:],signature.defaults):
			#print(k,v)
			outfile.write( str(k) + ' ' + str(v) +'\n')
			if i>2:
				args_dict[k] = v

	if updated_config:
		args_dict.update(updated_config)
	#print(args_dict)
	#print('config json', os.path.join(destination_directory,'env_configs.json'))
	with open(os.path.join(destination_directory,'env_configs.json'), "w") as outfile:
		json.dump(args_dict,outfile)

##########################################################################################################
# Stable baselines 
##########################################################################################################
def get_latest_model(path):
	""" Returns most recent model saved in path directory. """
	files = os.listdir(path)
	paths = [os.path.join(path, basename) for basename in files if basename.endswith('.zip')]
	return max(paths, key=os.path.getctime)

def read_env_config(directory):
	"""Read environment configuration from directory. """
	with open(os.path.join(directory,'env_configs.json')) as f:
		env_config = json.load(f)
	return env_config

# stable baselines 3
def get_sorted_dirs(path):
	files = os.listdir(path)
	print('files', files)
	dirs = [os.path.join(path,basename) for basename in files if os.path.isdir(os.path.join(path,basename))]
	print('dirs', dirs)
	# and basename is not '__pycache__'
	if '__pycache__' in dirs:
		dirs.remove('__pycache__')
	return sorted(dirs,key=os.path.getctime)

def load_all_results(path):
	"""Read all monitor files recursively, concatenate data together, plot. """
	#files = os.listdir(path)
	#gas_dirs = [os.path.join(path,basename) for basename in files if os.path.isdir(basename)]
	# get in order 0... N gas dirs
	#gas_dirs = sorted(gas_dirs,key=os.path.getctime)
	#
	tslist = []
	xaxis = 'timesteps'
	# get in order 0... N gas dirs
	gas_dirs = get_sorted_dirs(path)
	print('gas dirs', gas_dirs)
	# each gas dir may have multiple directories w monitor files, depending on performance
	for gas_dir in gas_dirs:
		#files = os.listdir(gas_dir)
		#trial_dirs = [os.path.join(gas_dir,basename) for basename in files if os.path.isdir(basename)]
		# sort
		#trial_dirs = sorted(trial_dirs,key=os.path.getctime)

		trial_dirs = get_sorted_dirs(gas_dir)

		print('trial dirs', trial_dirs)

		# extract monitor from each
		for trial_dir in trial_dirs:
			#data_frame = load_results(trial_dir)
			
			#for folder in dirs:
			timesteps = load_results(trial_dir)
			#if num_timesteps is not None:
			timesteps = timesteps[timesteps.l.cumsum() <= 10e10]
			tslist.append(timesteps)
			#plt.figure(1)
			# print('trial_dir', trial_dir)
			# print(timesteps)


	# iterate through and concatenate data 

	xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
	#print(xy_list)
	xy_list = concatenate_xy(xy_list)
	plot_curves(xy_list,xaxis, 'A1 Ep Rewards')
	plt.ylabel("Episode Rewards")

	xy_list = [ts2xy(timesteps_item, xaxis, True) for timesteps_item in tslist]
	xy_list = concatenate_xy(xy_list)
	plot_curves(xy_list, xaxis, 'A1 Ep Len')
	plt.ylabel("Episode Length")

def concatenate_xy(xy_list):
	curr_t = xy_list[0][0]
	curr_x = xy_list[0][1]
	new_xy = [np.array([0]),np.array([0])]
	for xy in xy_list:
		# print('xy', xy)
		# print('xy[0]', xy[0])
		# print('np.array(xy[0])', np.array(xy[0]))
		# print('new_xy', new_xy[0][-1])
		# print('test add', np.array(xy[0])+new_xy[0][-1])
		# print('new xy ', new_xy)
		# print('=======', np.array(new_xy[0]), np.array(xy[0])+new_xy[0][-1])
		# print('cocnat', np.concatenate((np.array(new_xy[0]), np.array(xy[0])+new_xy[0][-1])))
		new_xy[0] = np.concatenate((np.array(new_xy[0]), np.array(xy[0])+new_xy[0][-1]))
		new_xy[1] = np.concatenate((np.array(new_xy[1]), xy[1]))

	#print(tuple(new_xy))
	return [tuple(new_xy)]

##########################################################################################################
# rllib
##########################################################################################################
def get_latest_directory(path):
	""" Returns most recent directory in path. """
	files = os.listdir(path)
	# print('files', files)
	# for f in files:
	#     print(f, os.path.isdir(os.path.join(path, f)))
	paths = [os.path.join(path, f) for f in files if os.path.isdir(os.path.join(path, f))]
	#print('paths',paths)
	# __pycache__ files created when reading config files
	paths = [path for path in paths if '__pycache__' not in path]
	return max(paths, key=os.path.getctime)

def get_latest_model_rllib(path):
	""" Returns most recent model saved in path directory. """
	checkpoint = get_latest_directory(path)
	files = os.listdir(checkpoint)
	paths = [os.path.join(checkpoint, file) for file in files if ( file.startswith('checkpoint') and not file.endswith('.tune_metadata') )]
	print(paths)
	return paths[0] #max(paths, key=os.path.getctime)