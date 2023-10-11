#include "Gait.h"

Gait::Gait(int nMPC_segments, Vec4<int> offsets, Vec4<int> durations, const std::string& name) :
  _offsets(offsets.array()),
  _durations(durations.array()),
  _nIterations(nMPC_segments)
{
    _mpc_table = new int[nMPC_segments * 4];

    _offsetsPhase = offsets.cast<double>() / (double) nMPC_segments;
    _durationsPhase = durations.cast<double>() / (double) nMPC_segments;

    _stance = durations[0];
    _swing = nMPC_segments - durations[0];
}

Gait::~Gait()
{
    delete[] _mpc_table;
}

Vec4<double> Gait::getContactSubPhase()
{
  Array4d progress = _phase - _offsetsPhase;

  for(int i = 0; i < 4; i++)
  {
    if(progress[i] < 0) progress[i] += 1.;
    if(progress[i] > _durationsPhase[i])
    {
      progress[i] = 0.;
    }
    else
    {
      progress[i] = progress[i] / _durationsPhase[i];
    }
    
  }

  return progress.matrix();
}

Vec4<double> Gait::getSwingSubPhase()
{
  Array4d swing_offset = _offsetsPhase + _durationsPhase;
  for(int i = 0; i < 4; i++)
    if(swing_offset[i] > 1) swing_offset[i] -= 1.;
  Array4d swing_duration = 1. - _durationsPhase;

  Array4d progress = _phase - swing_offset;

  for(int i = 0; i < 4; i++)
  {
    if(progress[i] < 0) progress[i] += 1.;
    if(progress[i] > swing_duration[i])
    {
      progress[i] = 0.;
    }
    else
    {
      progress[i] = progress[i] / swing_duration[i];
    }
  }

  return progress.matrix();
}

int* Gait::mpc_gait()
{
  for(int i = 0; i < _nIterations; i++)
  {
    int iter = (i + _iteration) % _nIterations;
    Array4i progress = iter - _offsets;
    for(int j = 0; j < 4; j++)
    {
      if(progress[j] < 0) progress[j] += _nIterations;
      if(progress[j] < _durations[j])
        _mpc_table[i*4 + j] = 1;
      else
        _mpc_table[i*4 + j] = 0;
    }
  }
  //std::cout << "horizon: " << _iteration << std::endl;
  return _mpc_table;
}

void Gait::setIterations(int iterationsPerMPC, int currentIteration)
{
  _iteration = (currentIteration / iterationsPerMPC) % _nIterations;
  _phase = (double)(currentIteration % (iterationsPerMPC * _nIterations)) / (double) (iterationsPerMPC * _nIterations);
}
