#ifndef NOMINALMPC_H
#define NOMINALMPC_H

#include "MPCLocoMotion.h"
#include "footReactionPolicy.h"
#include "MPCSolver.h"

class NominalMPC : public MPCLocomotion {
public:
  NominalMPC(double _dt, int _iterations_between_mpc, int horizon);
  void run(ControlFSMData& data);
  void updateMPCIfNeeded(int* mpcTable, ControlFSMData& data);

  MPCSolver solver;
};


#endif 