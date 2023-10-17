# RL_augmented_MPC
Software implementation of RL-augmented MPC for quadruped locomotion. See experiment video(https://youtu.be/HxSIxTnEw08)
In this repo, two setups for RL-augmented MPC are implemented. One is foot placement offset to the Raiber heuristic and the other is joint angle offsets to the nominal swing trajectory (reported in the paper). Both setups are deployable to the robot and achieve similar performances.

## File Structure
- [training_module](./traning_module/) Pybullet training environment for the project
- [experiment_code](./experiment_code/) Experiment code to run the trained policies

Note: The training module doesn't interface directly with hardware !!!

## Dependencies
Go into README files of each folder for details

## TODO
- code cleanup
- documentation
- Credits and Acknowledges to be added later



