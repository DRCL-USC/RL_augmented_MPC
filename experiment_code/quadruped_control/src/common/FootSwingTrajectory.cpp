/*!
 * @file FootSwingTrajectory.cpp
 * @brief Utility to generate foot swing trajectories.
 *
 * Currently uses Bezier curves like Cheetah 3 does
 */

#include "../../include/common/Math/Interpolation.h"
#include "../../include/common/FootSwingTrajectory.h"

/*!
 * Compute foot swing trajectory with a bezier curve
 * @param phase : How far along we are in the swing (0 to 1)
 * @param swingTime : How long the swing should take (seconds)
 */

template <typename T>
void FootSwingTrajectory<T>::computeSwingTrajectoryBezier(T phase, T swingTime) {
  _p = Interpolate::cubicBezier<Vec3<T>>(_p0, _pf, phase);
  _v = Interpolate::cubicBezierFirstDerivative<Vec3<T>>(_p0, _pf, phase)/swingTime;
  // T phasePI = 2 * M_PI * phase;
  // _p = _p0 + (_pf - _p0) * (phasePI - sin(phasePI))/(2*M_PI);
  // _v = (_pf - _p0) * (1 - cos(phasePI))/swingTime;
  T zp, zv;

  // zp = _height*( 1 - cos(phasePI))/2 + _p0[2];
  // zv = _height * M_PI * sin(phasePI)/swingTime;
  if(phase < T(0.5)) {
    zp = Interpolate::cubicBezier<T>(_p0[2], _p0[2] + _height, phase * 2);
    zv = Interpolate::cubicBezierFirstDerivative<T>(_p0[2], _p0[2] + _height, phase * 2)/(swingTime*0.5);
  
  }
  else {
    zp = Interpolate::cubicBezier<T>(_p0[2] + _height, _pf[2], phase * 2 - 1);
    zv = Interpolate::cubicBezierFirstDerivative<T>(_p0[2] + _height, _pf[2], phase * 2 - 1)/(swingTime*0.5);
  }

  _p[2] = zp;
  _v[2] = zv;
 
}

template class FootSwingTrajectory<double>;
template class FootSwingTrajectory<float>;