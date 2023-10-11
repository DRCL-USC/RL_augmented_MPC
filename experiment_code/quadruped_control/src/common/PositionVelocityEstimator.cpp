#include "../../include/common/PositionVelocityEstimator.h"

void LinearKFPositionVelocityEstimator::setup() {
  double dt = 0.001;
  _xhat.setZero();
  _xhat[2] = 0.0;
  _ps.setZero();
  _vs.setZero();
  _A.setZero();
  _A.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(0, 3, 3, 3) = dt * Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(3, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(6, 6, 12, 12) = Eigen::Matrix<double, 12, 12>::Identity();
  _B.setZero();
  _B.block(3, 0, 3, 3) = dt * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C1(3, 6);
  C1 << Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C2(3, 6);
  C2 << Eigen::Matrix<double, 3, 3>::Zero(), Eigen::Matrix<double, 3, 3>::Identity();
  _C.setZero();
  _C.block(0, 0, 3, 6) = C1;
  _C.block(3, 0, 3, 6) = C1;
  _C.block(6, 0, 3, 6) = C1;
  _C.block(9, 0, 3, 6) = C1;
  _C.block(0, 6, 12, 12) = -1 * Eigen::Matrix<double, 12, 12>::Identity();
  _C.block(12, 0, 3, 6) = C2;
  _C.block(15, 0, 3, 6) = C2;
  _C.block(18, 0, 3, 6) = C2;
  _C.block(21, 0, 3, 6) = C2;
  _C(27, 17) = 1;
  _C(26, 14) = 1;
  _C(25, 11) = 1;
  _C(24, 8) = 1;
  _P.setIdentity();
  _P = 100 * _P;
  _Q0.setIdentity();
  _Q0.block(0, 0, 3, 3) = (dt / 20.f) * Eigen::Matrix<double, 3, 3>::Identity();
  _Q0.block(3, 3, 3, 3) =
      (dt * 9.8f / 20.f) * Eigen::Matrix<double, 3, 3>::Identity();
  _Q0.block(6, 6, 12, 12) = dt * Eigen::Matrix<double, 12, 12>::Identity();
  _R0.setIdentity();
  std::cout << "PosVel setup" << std::endl;
}

LinearKFPositionVelocityEstimator::LinearKFPositionVelocityEstimator() {}
 
// run estimator
void LinearKFPositionVelocityEstimator::run() {
  //std::cout << "run LinearKFPosVelEstimate" << std::endl;
  double process_noise_pimu = 0.02;
  double process_noise_vimu = 0.02;
  double process_noise_pfoot = 0.002;
  double sensor_noise_pimu_rel_foot = 0.001;
  double sensor_noise_vimu_rel_foot = 0.1;
  double sensor_noise_zfoot = 0.001;

  Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Identity();
  Q.block(0, 0, 3, 3) = _Q0.block(0, 0, 3, 3) * process_noise_pimu;
  Q.block(3, 3, 3, 3) = _Q0.block(3, 3, 3, 3) * process_noise_vimu;
  Q.block(6, 6, 12, 12) = _Q0.block(6, 6, 12, 12) * process_noise_pfoot;

  Eigen::Matrix<double, 28, 28> R = Eigen::Matrix<double, 28, 28>::Identity();
  R.block(0, 0, 12, 12) = _R0.block(0, 0, 12, 12) * sensor_noise_pimu_rel_foot;
  R.block(12, 12, 12, 12) =
      _R0.block(12, 12, 12, 12) * sensor_noise_vimu_rel_foot;
  R.block(24, 24, 4, 4) = _R0.block(24, 24, 4, 4) * sensor_noise_zfoot;

  int qindex = 0;
  int rindex1 = 0;
  int rindex2 = 0;
  int rindex3 = 0;

  // std::cout << "a_x " << this->_stateEstimatorData.result->aWorld << std::endl;

  Vec3<double> g(0, 0, -9.81);
  Mat3<double> Rbod = this->_stateEstimatorData.result->rBody.transpose();
  Vec3<double> a = this->_stateEstimatorData.result->aWorld + g;  // in old code, Rbod * se_acc + g
  // std::cout << "A WORLD\n" << a << "\n";
  Vec4<double> pzs = Vec4<double>::Zero();
  Vec4<double> trusts = Vec4<double>::Zero();
  Vec3<double> p0, v0;
  p0 << _xhat[0], _xhat[1], _xhat[2];
  v0 << _xhat[3], _xhat[4], _xhat[5];

  for (int i = 0; i < 4; i++) {
    int i1 = 3 * i;
    //std::cout << "loop: " << i << std::endl;
    //std::cout << "find hip location" << std::endl;
     Quadruped& quadruped =
        *(this->_stateEstimatorData.legControllerData->aliengo);

    //std::cout << "dynamics defined" << std::endl;
    Vec3<double> ph = quadruped.getHipLocation(i);  // hip positions relative to CoM

    //std::cout << ph << std::endl;

    Vec3<double> p_rel = ph + this->_stateEstimatorData.legControllerData[i].p;
    Vec3<double> dp_rel = this->_stateEstimatorData.legControllerData[i].v; 
    Vec3<double> p_f = Rbod * p_rel;
    Vec3<double> dp_f =
        Rbod *
        (this->_stateEstimatorData.result->omegaBody.cross(p_rel) + dp_rel);

    qindex = 6 + i1;
    rindex1 = i1;
    rindex2 = 12 + i1;
    rindex3 = 24 + i;

    double trust = 1;
    double phase = fmin(this->_stateEstimatorData.result->contactEstimate(i), 1);
    double trust_window = 0.3;

    if (phase < trust_window) {
      trust = phase / trust_window;
    } else if (phase > (1 - trust_window)) {
      trust = (1 - phase) / trust_window;
    }

    //printf("Trust %d: %.3f\n", i, trust);
    Q.block(qindex, qindex, 3, 3) =
        (1 + (1 - trust) * 100.0) * Q.block(qindex, qindex, 3, 3);
    R.block(rindex1, rindex1, 3, 3) = 1 * R.block(rindex1, rindex1, 3, 3);
    R.block(rindex2, rindex2, 3, 3) =
        (1 + (1 - trust) * 100.0) * R.block(rindex2, rindex2, 3, 3);
    R(rindex3, rindex3) =
        (1 + (1 - trust) * 100.0) * R(rindex3, rindex3);

    trusts(i) = trust;

    _ps.segment(i1, 3) = -p_f;
    _vs.segment(i1, 3) = (1.0f - trust) * v0 + trust * (-dp_f);
    pzs(i) = (1.0f - trust) * (p0(2) + p_f(2));
  }
  // std::cout << "_ps: " << std::endl;
  // std::cout << _ps << std::endl;
  // std::cout << "_vs: " << std::endl;
  // std::cout << _vs << std::endl;
  // std::cout << "pzs: " << std::endl;
  // std::cout << pzs << std::endl;

  Eigen::Matrix<double, 28, 1> y;
  y << _ps, _vs, pzs;
  _xhat = _A * _xhat + _B * a;
  Eigen::Matrix<double, 18, 18> At = _A.transpose();
  Eigen::Matrix<double, 18, 18> Pm = _A * _P * At + Q;
  Eigen::Matrix<double, 18, 28> Ct = _C.transpose();
  Eigen::Matrix<double, 28, 1> yModel = _C * _xhat;
  Eigen::Matrix<double, 28, 1> ey = y - yModel;
  Eigen::Matrix<double, 28, 28> S = _C * Pm * Ct + R;

  // todo compute LU only once
  Eigen::Matrix<double, 28, 1> S_ey = S.lu().solve(ey);
  _xhat += Pm * Ct * S_ey;

  //   std::cout << "_A: " << std::endl;
  // std::cout << _A << std::endl;
  // std::cout << "Pm: " << std::endl;
  // std::cout << Pm << std::endl;


  Eigen::Matrix<double, 28, 18> S_C = S.lu().solve(_C);
  _P = (Eigen::Matrix<double, 18, 18>::Identity() - Pm * Ct * S_C) * Pm;

  Eigen::Matrix<double, 18, 18> Pt = _P.transpose();
  _P = (_P + Pt) / 2;

  if (_P.block(0, 0, 2, 2).determinant() > 0.000001) {
    _P.block(0, 2, 2, 16).setZero();
    _P.block(2, 0, 16, 2).setZero();
    _P.block(0, 0, 2, 2) /= 10;
  }
  
  this->_stateEstimatorData.result->position = _xhat.block(0, 0, 3, 1);
  this->_stateEstimatorData.result->vWorld = _xhat.block(3, 0, 3, 1);
  this->_stateEstimatorData.result->vBody =
      this->_stateEstimatorData.result->rBody *
      this->_stateEstimatorData.result->vWorld;
}

// void CheaterPositionVelocityEstimator::run() {
//  // std::cout << "run StateEstimator" << std::endl;
//   this->_stateEstimatorData.result->position[0] = lowState.cheat.position[0];
//   this->_stateEstimatorData.result->position[1] = lowState.cheat.position[1];
//   this->_stateEstimatorData.result->position[2] = lowState.cheat.position[2];

//   this->_stateEstimatorData.result->vWorld[0] = lowState.cheat.vWorld[0];
//   this->_stateEstimatorData.result->vWorld[1] = lowState.cheat.vWorld[1];
//   this->_stateEstimatorData.result->vWorld[2] = lowState.cheat.vWorld[2];

//   this->_stateEstimatorData.result->vBody=
//     this->_stateEstimatorData.result->rBody * this->_stateEstimatorData.result->vWorld;
// }

TunedKFPositionVelocityEstimator::TunedKFPositionVelocityEstimator() {}

void TunedKFPositionVelocityEstimator::setup() {
  double dt = 0.001;
  _xhat.setZero();
  _xhat[2] = 0.0;
  _ps.setZero();
  _vs.setZero();
  _A.setZero();
  _A.block(0, 0, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(0, 3, 3, 3) = dt * Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(3, 3, 3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  _A.block(6, 6, 12, 12) = Eigen::Matrix<double, 12, 12>::Identity();
  _B.setZero();
  _B.block(3, 0, 3, 3) = dt * Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C1(3, 6);
  C1 << Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C2(3, 6);
  C2 << Eigen::Matrix<double, 3, 3>::Zero(), Eigen::Matrix<double, 3, 3>::Identity();
  _C.setZero();
  _C.block(0, 0, 3, 6) = C1;
  _C.block(3, 0, 3, 6) = C1;
  _C.block(6, 0, 3, 6) = C1;
  _C.block(9, 0, 3, 6) = C1;
  _C.block(0, 6, 12, 12) = -1 * Eigen::Matrix<double, 12, 12>::Identity();
  _C.block(12, 0, 3, 6) = C2;
  _C.block(15, 0, 3, 6) = C2;
  _C.block(18, 0, 3, 6) = C2;
  _C.block(21, 0, 3, 6) = C2;
  _C(27, 17) = 1;
  _C(26, 14) = 1;
  _C(25, 11) = 1;
  _C(24, 8) = 1;
  _P.setIdentity();
  _P = 100 * _P;

  _RInit <<  0.008 , 0.012 ,-0.000 ,-0.009 , 0.012 , 0.000 , 0.009 ,-0.009 ,-0.000 ,-0.009 ,-0.009 , 0.000 ,-0.000 , 0.000 ,-0.000 , 0.000 ,-0.000 ,-0.001 ,-0.002 , 0.000 ,-0.000 ,-0.003 ,-0.000 ,-0.001 , 0.000 , 0.000 , 0.000 , 0.000,
               0.012 , 0.019 ,-0.001 ,-0.014 , 0.018 ,-0.000 , 0.014 ,-0.013 ,-0.000 ,-0.014 ,-0.014 , 0.001 ,-0.001 , 0.001 ,-0.001 , 0.000 , 0.000 ,-0.001 ,-0.003 , 0.000 ,-0.001 ,-0.004 ,-0.000 ,-0.001 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.000, -0.001,  0.001,  0.001, -0.001,  0.000, -0.000,  0.000, -0.000,  0.001,  0.000, -0.000,  0.000, -0.000,  0.000,  0.000, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,
               -0.009, -0.014,  0.001,  0.010, -0.013,  0.000, -0.010,  0.010,  0.000,  0.010,  0.010, -0.000,  0.001,  0.000,  0.000,  0.001, -0.000,  0.001,  0.002, -0.000,  0.000,  0.003,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000,
               0.012 , 0.018 ,-0.001 ,-0.013 , 0.018 ,-0.000 , 0.013 ,-0.013 ,-0.000 ,-0.013 ,-0.013 , 0.001 ,-0.001 , 0.000 ,-0.001 , 0.000 , 0.001 ,-0.001 ,-0.003 , 0.000 ,-0.001 ,-0.004 ,-0.000 ,-0.001 , 0.000 , 0.000 , 0.000 , 0.000,
               0.000 ,-0.000 , 0.000 , 0.000 ,-0.000 , 0.001 , 0.000 , 0.000 ,-0.000 , 0.000 , 0.000 ,-0.000 ,-0.000 , 0.000 ,-0.000 , 0.000 , 0.000 , 0.000 ,-0.000 ,-0.000 ,-0.000 ,-0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000,
               0.009 , 0.014 ,-0.000 ,-0.010 , 0.013 , 0.000 , 0.010 ,-0.010 ,-0.000 ,-0.010 ,-0.010 , 0.000 ,-0.001 , 0.000 ,-0.001 , 0.000 ,-0.000 ,-0.001 ,-0.001 , 0.000 ,-0.000 ,-0.003 ,-0.000 ,-0.001 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.009, -0.013,  0.000,  0.010, -0.013,  0.000, -0.010,  0.009,  0.000,  0.010,  0.010, -0.000,  0.001, -0.000,  0.000, -0.000,  0.000,  0.001,  0.002,  0.000,  0.000,  0.003,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000,
               -0.000, -0.000, -0.000,  0.000, -0.000, -0.000, -0.000,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000, -0.000,  0.000, -0.000,  0.000, -0.000,  0.000, -0.000,  0.000,  0.000, -0.000, -0.000,  0.000,  0.000,  0.000,  0.000,
               -0.009, -0.014,  0.001,  0.010, -0.013,  0.000, -0.010,  0.010,  0.000,  0.010,  0.010, -0.000,  0.001,  0.000,  0.000, -0.000, -0.000,  0.001,  0.002, -0.000,  0.000,  0.003,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000,
               -0.009, -0.014,  0.000,  0.010, -0.013,  0.000, -0.010,  0.010,  0.000,  0.010,  0.010, -0.000,  0.001, -0.000,  0.000, -0.000,  0.000,  0.001,  0.002, -0.000,  0.000,  0.003,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,
               0.000 , 0.001 ,-0.000 ,-0.000 , 0.001 ,-0.000 , 0.000 ,-0.000 , 0.000 ,-0.000 ,-0.000 , 0.001 , 0.000 ,-0.000 ,-0.000 ,-0.000 , 0.000 , 0.000 ,-0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.000, -0.001,  0.000,  0.001, -0.001, -0.000, -0.001,  0.001,  0.000,  0.001,  0.001,  0.000,  1.708,  0.048,  0.784,  0.062,  0.042,  0.053,  0.077,  0.001, -0.061,  0.046, -0.019, -0.029,  0.000,  0.000,  0.000,  0.000,
               0.000 , 0.001 ,-0.000 , 0.000 , 0.000 , 0.000 , 0.000 ,-0.000 ,-0.000 , 0.000 ,-0.000 ,-0.000 , 0.048 , 5.001 ,-1.631 ,-0.036 , 0.144 , 0.040 , 0.036 , 0.016 ,-0.051 ,-0.067 ,-0.024 ,-0.005 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.000, -0.001,  0.000,  0.000, -0.001, -0.000, -0.001,  0.000,  0.000,  0.000,  0.000, -0.000,  0.784, -1.631,  1.242,  0.057, -0.037,  0.018,  0.034, -0.017, -0.015,  0.058, -0.021, -0.029,  0.000,  0.000,  0.000,  0.000,
               0.000 , 0.000 , 0.000 , 0.001 , 0.000 , 0.000 , 0.000 ,-0.000 ,-0.000 ,-0.000 ,-0.000 ,-0.000 , 0.062 ,-0.036 , 0.057 , 6.228 ,-0.014 , 0.932 , 0.059 , 0.053 ,-0.069 , 0.148 , 0.015 ,-0.031 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.000,  0.000, -0.000, -0.000,  0.001,  0.000, -0.000,  0.000,  0.000, -0.000,  0.000,  0.000,  0.042,  0.144, -0.037, -0.014,  3.011,  0.986,  0.076,  0.030, -0.052, -0.027,  0.057,  0.051,  0.000,  0.000,  0.000,  0.000,
               -0.001, -0.001, -0.000,  0.001, -0.001,  0.000, -0.001,  0.001, -0.000,  0.001,  0.001,  0.000,  0.053,  0.040,  0.018,  0.932,  0.986,  0.885,  0.090,  0.044, -0.055,  0.057,  0.051, -0.003,  0.000,  0.000,  0.000,  0.000,
               -0.002, -0.003,  0.000,  0.002, -0.003, -0.000, -0.001,  0.002,  0.000,  0.002,  0.002, -0.000,  0.077,  0.036,  0.034,  0.059,  0.076,  0.090,  6.230,  0.139,  0.763,  0.013, -0.019, -0.024,  0.000,  0.000,  0.000,  0.000,
               0.000 , 0.000 ,-0.000 ,-0.000 , 0.000 ,-0.000 , 0.000 , 0.000 ,-0.000 ,-0.000 ,-0.000 , 0.000 , 0.001 , 0.016 ,-0.017 , 0.053 , 0.030 , 0.044 , 0.139 , 3.130 ,-1.128 ,-0.010 , 0.131 , 0.018 , 0.000 , 0.000 , 0.000 , 0.000,
               -0.000, -0.001, -0.000,  0.000, -0.001, -0.000, -0.000,  0.000,  0.000,  0.000,  0.000,  0.000, -0.061, -0.051, -0.015, -0.069, -0.052, -0.055,  0.763, -1.128,  0.866, -0.022, -0.053,  0.007,  0.000,  0.000,  0.000,  0.000,
               -0.003, -0.004, -0.000,  0.003, -0.004, -0.000, -0.003,  0.003,  0.000,  0.003,  0.003,  0.000,  0.046, -0.067,  0.058,  0.148, -0.027,  0.057,  0.013, -0.010, -0.022,  2.437, -0.102,  0.938,  0.000,  0.000,  0.000,  0.000,
               -0.000, -0.000,  0.000,  0.000, -0.000,  0.000, -0.000,  0.000, -0.000,  0.000,  0.001,  0.000, -0.019, -0.024, -0.021,  0.015,  0.057,  0.051, -0.019,  0.131, -0.053, -0.102,  4.944,  1.724,  0.000,  0.000,  0.000,  0.000,
               -0.001, -0.001,  0.000,  0.001, -0.001,  0.000, -0.001,  0.001, -0.000,  0.001,  0.001,  0.000, -0.029, -0.005, -0.029, -0.031,  0.051, -0.003, -0.024,  0.018,  0.007,  0.938,  1.724,  1.569,  0.000,  0.000,  0.000,  0.000,
               0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 1.000 , 0.000 , 0.000 , 0.000,
               0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 1.000 , 0.000 , 0.000,
               0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 1.000 , 0.000,
               0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 0.000 , 1.000;

    _Cu <<   268.573,  -43.819, -147.211,
            -43.819 ,  92.949 ,  58.082,
            -147.211,   58.082,  302.120;

    for(int i(0); i<_Qdig.rows(); ++i){
        if(i < 6){
            _Qdig(i) = 0.0003;
        }else{
            _Qdig(i) = 0.01;
        }
    }

    _QInit = _Qdig.asDiagonal();
    _QInit +=  _B * _Cu * _B.transpose();
  
  std::cout << "PosVel setup" << std::endl;  
}

void TunedKFPositionVelocityEstimator::run() {
  //std::cout << "run LinearKFPosVelEstimate" << std::endl;
  _Q = _QInit;
  _R = _RInit;

  int qindex = 0;
  int rindex1 = 0;
  int rindex2 = 0;
  int rindex3 = 0;

  // std::cout << "a_x " << this->_stateEstimatorData.result->aWorld << std::endl;

  Vec3<double> g(0, 0, -9.81);
  Mat3<double> Rbod = this->_stateEstimatorData.result->rBody.transpose();
  Vec3<double> a = this->_stateEstimatorData.result->aWorld + g;  // in old code, Rbod * se_acc + g
  // std::cout << "A WORLD\n" << a << "\n";
  Vec4<double> pzs = Vec4<double>::Zero();
  Vec4<double> trusts = Vec4<double>::Zero();
  Vec3<double> p0, v0;
  p0 << _xhat[0], _xhat[1], _xhat[2];
  v0 << _xhat[3], _xhat[4], _xhat[5];

  for (int i = 0; i < 4; i++) {
    int i1 = 3 * i;
    //std::cout << "loop: " << i << std::endl;
    //std::cout << "find hip location" << std::endl;
     Quadruped& quadruped =
        *(this->_stateEstimatorData.legControllerData->aliengo);

    //std::cout << "dynamics defined" << std::endl;
    Vec3<double> ph = quadruped.getHipLocation(i);  // hip positions relative to CoM

    //std::cout << ph << std::endl;

    Vec3<double> p_rel = ph + this->_stateEstimatorData.legControllerData[i].p;
    Vec3<double> dp_rel = this->_stateEstimatorData.legControllerData[i].v; 
    Vec3<double> p_f = Rbod * p_rel;
    Vec3<double> dp_f =
        Rbod *
        (this->_stateEstimatorData.result->omegaBody.cross(p_rel) + dp_rel);

    qindex = 6 + i1;
    rindex1 = i1;
    rindex2 = 12 + i1;
    rindex3 = 24 + i;

    double trust = 1;
    double phase = fmin(this->_stateEstimatorData.result->contactEstimate(i), 1);
    double trust_window = 0.3;

    if (phase < trust_window) {
      trust = phase / trust_window;
    } else if (phase > (1 - trust_window)) {
      trust = (1 - phase) / trust_window;
    }

    //printf("Trust %d: %.3f\n", i, trust);
    _Q.block(qindex, qindex, 3, 3) =
        (1 + (1 - trust) * 100.0) * _Q.block(qindex, qindex, 3, 3);
    _R.block(rindex1, rindex1, 3, 3) = 1 * _R.block(rindex1, rindex1, 3, 3);
    _R.block(rindex2, rindex2, 3, 3) =
        (1 + (1 - trust) * 100.0) * _R.block(rindex2, rindex2, 3, 3);
    _R(rindex3, rindex3) =
        (1 + (1 - trust) * 100.0) * _R(rindex3, rindex3);

    trusts(i) = trust;

    _ps.segment(i1, 3) = -p_f;
    _vs.segment(i1, 3) = (1.0f - trust) * v0 + trust * (-dp_f);
    pzs(i) = (1.0f - trust) * (p0(2) + p_f(2));
  }
  // std::cout << "_ps: " << std::endl;
  // std::cout << _ps << std::endl;
  // std::cout << "_vs: " << std::endl;
  // std::cout << _vs << std::endl;
  // std::cout << "pzs: " << std::endl;
  // std::cout << pzs << std::endl;

  Eigen::Matrix<double, 28, 1> y;
  y << _ps, _vs, pzs;
  _xhat = _A * _xhat + _B * a;
  Eigen::Matrix<double, 18, 18> At = _A.transpose();
  Eigen::Matrix<double, 18, 18> Pm = _A * _P * At + _Q;
  Eigen::Matrix<double, 18, 28> Ct = _C.transpose();
  Eigen::Matrix<double, 28, 1> yModel = _C * _xhat;
  Eigen::Matrix<double, 28, 1> ey = y - yModel;
  Eigen::Matrix<double, 28, 28> S = _C * Pm * Ct + _R;

  // todo compute LU only once
  Eigen::Matrix<double, 28, 1> S_ey = S.lu().solve(ey);
  _xhat += Pm * Ct * S_ey;

  //   std::cout << "_A: " << std::endl;
  // std::cout << _A << std::endl;
  // std::cout << "Pm: " << std::endl;
  // std::cout << Pm << std::endl;


  Eigen::Matrix<double, 28, 18> S_C = S.lu().solve(_C);
  _P = (Eigen::Matrix<double, 18, 18>::Identity() - Pm * Ct * S_C) * Pm;

  Eigen::Matrix<double, 18, 18> Pt = _P.transpose();
  _P = (_P + Pt) / 2;

  if (_P.block(0, 0, 2, 2).determinant() > 0.000001) {
    _P.block(0, 2, 2, 16).setZero();
    _P.block(2, 0, 16, 2).setZero();
    _P.block(0, 0, 2, 2) /= 10;
  }
  
  this->_stateEstimatorData.result->position = _xhat.block(0, 0, 3, 1);
  this->_stateEstimatorData.result->vWorld = _xhat.block(3, 0, 3, 1);
  this->_stateEstimatorData.result->vBody =
      this->_stateEstimatorData.result->rBody *
      this->_stateEstimatorData.result->vWorld;
}