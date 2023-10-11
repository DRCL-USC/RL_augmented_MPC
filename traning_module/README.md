# Learning
Cleaned up learning codebase for quadruped robot learning

## Installation

Then, install usc_learning as a package:

`python3 setup.py install --user `
or 
`pip3 install -e . `


## Code structure

- [learning] training and testing scripts
    *`learning/rllib_helpers` stores policy setup
    *`learning/train_mpc.py` train the policy, switch between different envs in here
    *`learning/test_mlp.py` test the learnt mlp policy and saves the policy into a .json file for hardware experiment
- [envs] robot class and task environment for trainning
    *`envs/A1.py` the robot interface with pybullet (a simplified version of robot class from motion imitation library (https://github.com/erwincoumans/motion_imitation))

    *`envs/MPC_*` training envrionment of the RL policies to augment MPC for foot reaction, uncertaintiy adaptiaton and agile locomotion. The `MPCLocomotion` object is called in these envrionment files to update GRF. In the `step` function, dynamics compensation and swing trajecotry offset are added to the MPC controller.

- [MPC_implementation] the implementation of MPC controller for quadruped robot
    *`mpc_implementation/mpc_osqp.cc` A modified solver from motion imitation library (https://github.com/erwincoumans/motion_imitation) that tailor to the proposed fromulation
    *`mpc_implementation/MPCLocomotion.py` runs the MPC optimization at desired frequency while updating the swing trajectory for swing foot. Everything is transformed into world/odometry frame to feed to the solver similar to Cheetah-Software (https://github.com/dbdxnuliba/mit-biomimetics_Cheetah). 

## Train a policy
-`cd learning`
-`python3 train_mpc.py` 

## Test the policy
-`cd learning`
-`python3 test_mlp.py` 
