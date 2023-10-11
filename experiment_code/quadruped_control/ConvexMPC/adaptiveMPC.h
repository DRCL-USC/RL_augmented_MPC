#ifndef adapMPC_H
#define adapMPC_H

#include "MPCLocoMotion.h"
#include "adaptivePolicy.h"
#include "MPCSolver.h"

class adaptiveMPC : public MPCLocomotion {
public:
  adaptiveMPC(double _dt, int _iterations_between_mpc, int horizon);

  void run(ControlFSMData& data);

  void updateMPCIfNeeded(int* mpcTable, ControlFSMData& data);

  MPCSolver solver;
  adaptivePolicy adap_policy;

  double foothold_heuristic[8];
  double foothold_offset[8];
  Eigen::Vector4d contact;

};

#endif 