#ifndef rlMPC_H
#define rlMPC_H

#include "MPCLocoMotion.h"
#include "footReactionPolicy.h"
#include "adaptivePolicy.h"
#include "MPCSolver.h"

class rlMPC : public MPCLocomotion {
public:
  rlMPC(double _dt, int _iterations_between_mpc, int horizon);

  void run(ControlFSMData& data);

  void updateMPCIfNeeded(int* mpcTable, ControlFSMData& data);

  MPCSolver solver;
  footReactionPolicy policy;
  // adaptivePolicy policy;

  double foothold_heuristic[8];
  Eigen::Vector4d contact;

};

#endif 