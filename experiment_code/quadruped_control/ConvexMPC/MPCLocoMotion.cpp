#include "MPCLocoMotion.h"

MPCLocomotion::MPCLocomotion(double _dt, int _iterations_between_mpc, int horizon):
  iterationsBetweenMPC(_iterations_between_mpc),
  horizonLength(horizon),
  dt(_dt),
  galloping(horizonLength, Vec4<int>(0,2,7,9),Vec4<int>(5,5,5,5),"Galloping"),
  pronking(horizonLength, Vec4<int>(0,0,0,0),Vec4<int>(4,4,4,4),"Pronking"),
  trotting(horizonLength, Vec4<int>(0,5,5,0), Vec4<int>(5,5,5,5),"trotting"),
  standing(horizonLength, Vec4<int>(0,0,0,0), Vec4<int>(10,10,10,10),"Standing"),
  bounding(horizonLength, Vec4<int>(5,5,0,0),Vec4<int>(5,5,5,5),"Bounding"),
  walking(horizonLength, Vec4<int>(0,3,5,8), Vec4<int>(5,5,5,5), "Walking"),
  pacing(horizonLength, Vec4<int>(5,0,5,0),Vec4<int>(5,5,5,5),"Pacing"),
  two_foot_trot(horizonLength, Vec4<int>(0,0,5,0), Vec4<int>(10,10,5,5), "two_foot_trot"),
  flying_trot(10, Vec4<int>(0,5,5,0), Vec4<int>(3,3,3,3), "flying_trot")
{
  dtMPC = dt * iterationsBetweenMPC;
  //std::cout << "dtMPC: " << dtMPC << std::endl;
  rpy_int[2] = 0;
  for(int i = 0; i < 4; i++)
    firstSwing[i] = true;

  // foot_position.open("foot_pos.txt");
}
