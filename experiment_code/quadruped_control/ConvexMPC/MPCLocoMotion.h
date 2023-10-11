#ifndef MPCLOCOMOTION_H
#define MPCLOCOMOTION_H

#include <fstream>
#include <stdio.h>
#include <iostream>
#include "../include/common/Utilities/Timer.h"
#include "../include/common/Math/orientation_tools.h"
#include "../include/common/cppTypes.h"
#include "../include/common/FootSwingTrajectory.h"
#include "../include/common/ControlFSMData.h"
#include "Gait.h"
// #include "MPCSolver.h"

using namespace std;

using Eigen::Array4f;
using Eigen::Array4i;
using Eigen::Array4d;

class MPCLocomotion{
public:
  MPCLocomotion(double _dt, int _iterations_between_mpc, int horizon);
  void setGaitNum(int gaitNum){gaitNumber = gaitNum; return;}
  virtual void run(ControlFSMData& data) = 0;

  Vec4<double> contact_state;
  bool firstRun = true;
  bool climb = 0;
  // ofstream foot_position;
  double phase;
  double t = 0.0;

protected:
  int iterationsBetweenMPC;
  int horizonLength;
  double dt;
  double dtMPC;
  int iterationCounter = 0;
  Vec3<double> f_ff[4];
  Vec4<double> swingTimes;
  FootSwingTrajectory<double> footSwingTrajectories[4];
  Gait trotting, bounding, pacing, walking, galloping, pronking, standing, two_foot_trot, flying_trot;
  Mat3<double> Kp, Kd, Kp_stance, Kd_stance;

  bool firstSwing[4];
  double swingTimeRemaining[4];
  double stand_traj[6];
  double foothold_offset[8];
  int current_gait;
  int gaitNumber;

  Vec3<double> world_position_desired;
  Vec3<double> rpy_int;
  Vec3<double> rpy_comp;
  Vec3<double> pFoot[4];
  double trajAll[12*36];

  Mat43<double> W; // W = [1, px, py]
  Vec3<double> a; // a = [a_0, a_1, a_2]; z(x,y) = a_0 + a_1x + a_2y
  Vec4<double> pz;
  double ground_pitch;
};

#endif