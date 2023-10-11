## Quadruped_ROS_Simulation_V2
Software package for hardware experiment for RL-augmented MPC (A1, AlienGo)
    Note: Slight modifications are needed to accomandate Go1 and B1 robots

## Code structure for quadruped_control package
- [Common] includes leg controller, state Estimation, kinematic parameters and utility functions
- [Interface] IO interfaces for ROS/hardware
    * ROS: keyboard commands 
    * Hardware: joystick (under development)
- [Messages] messages for communication (corresponds with messages in unitree_legged_sdk)
- [FSM] finite state machine and FSM states to switch between controllers (passive, PDstand, QPstand, MPC locomotion)
- [ConvexMPC] the MPC controller. It contains the MPC solver and a base class to run the controller. Besides, the learnt policy is imported via `rapidJSON` package and the corresponding MPC controller class use the `policyIO` object to calcaulte the output from the policy. The current setup is fixed to a 3-layer MLP followered by a linear output layer, can be adapted for different setup later

## Dependencies
* [Boost](http://www.boost.org) (version 1.5.4 or higher)
* [CMake](http://www.cmake.org) (version 2.8.3 or higher)
* [LCM](https://lcm-proj.github.io) (version 1.4.0 or higher)
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) (>3.3)
* [rapidJSON](https://rapidjson.org/)


```

## Installation
 `mkdir build `
 `cd build`
 `cmake .. -DCMAKE_BUILD_TYPE=Release`
 `make`

## How to Run
* setup the network as in `unitree_legged_sdk` pacakge for A1 robot. The sdk inside this pacakage is compatible with A1 and AlienGo robots. For go1 robot, change the sdk

 `cd build`
 `sudo ./quad_ctrl`

## User Commands & Interface
* check the `Interface/KeyBoard.cpp` or `Interface/WirelessHandle.cpp` to see and modify definition of keyboard/joystick commands

## Todo
* Code cleanup
* Documentation


