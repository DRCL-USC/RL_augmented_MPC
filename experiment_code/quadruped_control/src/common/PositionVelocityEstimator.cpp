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


