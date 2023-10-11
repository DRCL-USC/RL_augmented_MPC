#ifndef GAIT
#define GAIT

#include "common_types.h"
#include "../include/common/cppTypes.h"
#include <fstream>
#include <stdio.h>

using namespace std;

using Eigen::Array4f;
using Eigen::Array4i;
using Eigen::Array4d;

class Gait
{
public:
  Gait(int nMPC_segments, Vec4<int> offsets, Vec4<int>  durations, const std::string& name="");
  ~Gait();
  Vec4<double> getContactSubPhase();
  Vec4<double> getSwingSubPhase();
  int* mpc_gait();
  void setIterations(int iterationsPerMPC, int currentIteration);
  int _stance;
  int _swing;
 double _phase;

private:
  int _nMPC_segments;
  int* _mpc_table;
  Array4i _offsets; // offset in mpc segments
  Array4i _durations; // duration of step in mpc segments
  Array4d _offsetsPhase; // offsets in phase (0 to 1)
  Array4d _durationsPhase; // durations in phase (0 to 1)
  int _iteration;
  int _nIterations;

};

#endif