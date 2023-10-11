#include <iostream>
#include "NominalMPC.h"
#include "../include/common/Math/orientation_tools.h"

using namespace ori;

/* =========================== Controller ============================= */
NominalMPC::NominalMPC(double _dt, int _iterations_between_mpc, int horizon)
    : MPCLocomotion(_dt, _iterations_between_mpc, horizon),
      solver(_dt * _iterations_between_mpc, horizon, 0.2, 350) {}

void NominalMPC::run(ControlFSMData &data)
{
  auto &seResult = data._stateEstimator->getResult();
  auto &stateCommand = data._desiredStateCommand;
  // std::cout << "in side mpc" << seResult.rBody << std::endl;;

  // pick gait
  Gait *gait = &trotting;
  if (gaitNumber == 1)
    gait = &bounding;
  else if (gaitNumber == 2)
    gait = &trotting;
  else if (gaitNumber == 3)
    gait = &walking;
  else if (gaitNumber == 4)
    gait = &pacing;
  else if (gaitNumber == 5)
    gait = &galloping;
  else if (gaitNumber == 6)
    gait = &pronking;
  else if (gaitNumber == 7)
    gait = &standing;
  else if (gaitNumber == 8)
    gait = &two_foot_trot;

  current_gait = gaitNumber;
  // integrate position setpoint
  // get desired velocity from data.stateDes (in local coord)
  Vec3<double> v_des_robot(stateCommand->data.stateDes[6], stateCommand->data.stateDes[7], 0);
  Vec3<double> v_des_world;

  // convert desired linear vel from local coord into local coord
  v_des_world = seResult.rBody.transpose() * v_des_robot;
  Vec3<double> v_robot = seResult.vBody;

  world_position_desired[0] += dt * v_des_world[0];
  world_position_desired[1] += dt * v_des_world[1];
  // this should controlls desired robot height (in world coord), since we doing the opti under world coord
  // ununsed
  world_position_desired[2] = 0.3;

  // printf("p_des \t%.6f\n", dt * v_des_world[0]);
  // Integral-esque pitch and roll compensation
  if (fabs(v_robot[0]) > .2) // avoid dividing by zero
  {
    rpy_int[1] += dt * (stateCommand->data.stateDes[4] /*-hw_i->state_estimator->se_ground_pitch*/ - seResult.rpy[1]) / v_robot[0];
  }
  if (fabs(v_robot[1]) > 0.1)
  {
    rpy_int[0] += dt * (stateCommand->data.stateDes[3] /*-hw_i->state_estimator->se_ground_pitch*/
                        - seResult.rpy[0]) /
                  v_robot[1];
  }

  rpy_int[0] = fminf(fmaxf(rpy_int[0], -.25), .25);
  rpy_int[1] = fminf(fmaxf(rpy_int[1], -.25), .25);
  rpy_comp[1] = v_robot[0] * rpy_int[1];
  rpy_comp[0] = v_robot[1] * rpy_int[0]; // turn off for pronking

  for (int i = 0; i < 4; i++)
  {
    pFoot[i] = seResult.position + seResult.rBody.transpose() * (data._quadruped->getHipLocation(i) + data._legController->data[i].p);
  }

  // if(iterationCounter % 600 == 0){
  for (int i = 0; i < 4; i++)
  {
    W.row(i) << 1, footSwingTrajectories[i].getInitialPosition()[0], footSwingTrajectories[i].getInitialPosition()[1];
    pz[i] = footSwingTrajectories[i].getInitialPosition()[2];
    // if(i != 0) W.row(i) << 0, pFoot[i][0], pFoot[i][1];
  }
  a = W.transpose() * W * (W.transpose() * W * W.transpose() * W).inverse() * W.transpose() * pz;
  ground_pitch = acos(-1 / sqrt(a[1] * a[1] + a[2] * a[2] + 1)) - 3.14;
  // std::cout << "ground pitch: " << ground_pitch << std::endl;
  if (pz[0] < pz[2])
  {
    ground_pitch = -ground_pitch;
  }
  if (abs(pz[0] - pz[2]) < 0.01)
  {
    ground_pitch = 0;
  }
  // }

  // some first time initialization
  if (firstRun)
  {
    std::cout << "Run MPC" << std::endl;
    world_position_desired[0] = seResult.position[0];
    world_position_desired[1] = seResult.position[1];
    // init, desired z = z from state estimator, unused
    world_position_desired[2] = seResult.position[2];

    Vec3<double> v_des_robot(0, 0, 0); // connect to desired state command later

    Vec3<double> v_des_world(0, 0, 0); // connect to desired state command later

    Vec3<double> v_robot = seResult.vBody;
    iterationCounter = 0;
    for (int i = 0; i < 4; i++)
    {
      footSwingTrajectories[i].setHeight(0.1);
      footSwingTrajectories[i].setInitialPosition(pFoot[i]);
      footSwingTrajectories[i].setFinalPosition(pFoot[i]);
      // std::cout << "orig foot pos " << i << pFoot[i] << std::endl;
    }

    swingTimes[0] = dtMPC * gait->_swing;
    swingTimes[1] = dtMPC * gait->_swing;
    swingTimes[2] = dtMPC * gait->_swing;
    swingTimes[3] = dtMPC * gait->_swing;
    firstRun = false;
  }

  contact_state = gait->getContactSubPhase();
  double side_sign[4] = {-1, 1, -1, 1};
  double interleave_y[4] = {-0.08, 0.08, 0.01, -0.01};
  double interleave_gain = -0.2;
  double v_abs = std::fabs(seResult.vBody[0]);
  for (int i = 0; i < 4; i++)
  {

    if (firstSwing[i])
    {
      swingTimeRemaining[i] = swingTimes[i];
    }
    else
    {
      swingTimeRemaining[i] -= dt;
    }
    footSwingTrajectories[i].setHeight(.08);
    Vec3<double> offset(0.0, side_sign[i] * 0.0838, 0);

    Vec3<double> pRobotFrame = (data._quadruped->getHipLocation(i) + offset);
    Vec3<double> pYawCorrected = coordinateRotation(CoordinateAxis::Z, -stateCommand->data.stateDes[11] * gait->_stance * dtMPC / 2) * pRobotFrame;

    Vec3<double> des_vel;

    des_vel[0] = stateCommand->data.stateDes(6);
    des_vel[1] = stateCommand->data.stateDes(7);
    des_vel[2] = stateCommand->data.stateDes(8);

    Vec3<double> Pf = seResult.position +
                      seResult.rBody.transpose() * pYawCorrected + seResult.vWorld * swingTimeRemaining[i];

    //+ seResult.vWorld * swingTimeRemaining[i];

    double p_rel_max = 0.3;
    double pfx_rel = seResult.vWorld[0] * .5 * gait->_stance * dtMPC +
                     .03 * (seResult.vWorld[0] - v_des_world[0]) +
                     (0.5 * seResult.position[2] / 9.81) * (seResult.vWorld[1] * stateCommand->data.stateDes[11]);
    double pfy_rel = seResult.vWorld[1] * .5 * gait->_stance * dtMPC +
                     .03 * (seResult.vWorld[1] - v_des_world[1]) +
                     (0.5 * seResult.position[2] / 9.81) * (-seResult.vWorld[0] * stateCommand->data.stateDes[11]);
    pfx_rel = fminf(fmaxf(pfx_rel, -p_rel_max), p_rel_max);
    pfy_rel = fminf(fmaxf(pfy_rel, -p_rel_max), p_rel_max);
    Pf[0] += pfx_rel;
    Pf[1] += pfy_rel;
    // + interleave_y[i] * interleave_gain * v_abs;
    Pf[2] = 0.0;

    footSwingTrajectories[i].setFinalPosition(Pf);
    //}
  }

  // calc gait
  gait->setIterations(iterationsBetweenMPC, iterationCounter);
  int *mpcTable = gait->mpc_gait();
  phase = gait->_phase;

  // gait
  Vec4<double> contactStates = gait->getContactSubPhase();
  Vec4<double> swingStates = gait->getSwingSubPhase();

  Vec3<double> vDesLeg_stand(0, 0, 0);

  for (int foot = 0; foot < 4; foot++)
  {

    double contactState = contactStates(foot);
    double swingState = swingStates(foot);
    // std::cout << "swing" << foot << ": " << swingState << std::endl;
    if (swingState > 0) // foot is in swing
    {

      if (firstSwing[foot])
      {
        firstSwing[foot] = false;
      }
      // std::cout << "swing" << foot << ": " << swingState << std::endl;
      footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, swingTimes[foot]);

      Vec3<double> pDesFootWorld = footSwingTrajectories[foot].getPosition().cast<double>();
      Vec3<double> vDesFootWorld = footSwingTrajectories[foot].getVelocity().cast<double>();
      if(swingState > 0.5)
      {
       vDesFootWorld << 0, 0, 0;
      }

      Vec3<double> pDesLeg = seResult.rBody * (pDesFootWorld - seResult.position) - data._quadruped->getHipLocation(foot);
      Vec3<double> vDesLeg = seResult.rBody * (vDesFootWorld - seResult.vWorld);

      data._legController->commands[foot].feedforwardForce << 0, 0, 0;
      data._legController->commands[foot].pDes = pDesLeg;
      data._legController->commands[foot].vDes = vDesLeg;

      // data._legController->commands[foot].kpCartesian = Kp;
      // data._legController->commands[foot].kdCartesian = Kd;
      data._legController->commands[foot].kpJoint.diagonal() << 40, 40, 40;
      data._legController->commands[foot].kdJoint.diagonal() << 2, 2, 2;
      // account for early contact
      if (data._stateEstimator->getResult().contactEstimate(foot) != 0)
      {
        mpcTable[foot] = 1;
        //   // data._legController->commands[foot].feedforwardForce <<0, 0, -20;
      }
    }
    else if (contactState > 0 || data._stateEstimator->getResult().contactEstimate(foot) != 0) // foot is in stance
    {
      firstSwing[foot] = true;
      footSwingTrajectories[foot].setInitialPosition(pFoot[foot]);
      footSwingTrajectories[foot].computeSwingTrajectoryBezier(swingState, swingTimes[foot]);
      Vec3<double> pDesFootWorld = footSwingTrajectories[foot].getPosition().cast<double>();
      Vec3<double> vDesFootWorld = footSwingTrajectories[foot].getVelocity().cast<double>();
      Vec3<double> pDesLeg = seResult.rBody * (pDesFootWorld - seResult.position) - data._quadruped->getHipLocation(foot);
      Vec3<double> vDesLeg = seResult.rBody * (vDesFootWorld - seResult.vWorld);

      //  data._legController->commands[foot].kpCartesian = Kp_stance; // 0
      //  data._legController->commands[foot].kdCartesian = Kd_stance
      data._legController->commands[foot].kpJoint.diagonal() << 0, 0, 0;
      data._legController->commands[foot].kdJoint.diagonal() << 2, 2, 2;

      data._legController->commands[foot].feedforwardForce = f_ff[foot];

      if (gaitNumber == 8)
      {
        // std::cout << "two foot gait " << std::endl;
        for (int i = 0; i < 2; i++)
          data._legController->commands[i].vDes << 0, 0, 0;
      }
    }
  }

  updateMPCIfNeeded(mpcTable, data);

  iterationCounter++;
}

void NominalMPC::updateMPCIfNeeded(int *mpcTable, ControlFSMData &data)
{
  if ((iterationCounter % 25) == 0)
  {
    auto seResult = data._stateEstimator->getResult();
    auto &stateCommand = data._desiredStateCommand;

    double *p = seResult.position.data();
    double *v = seResult.vWorld.data();
    double *w = seResult.omegaWorld.data();
    double *q = seResult.orientation.data();

    double r[12];
    for (int i = 0; i < 12; i++)
      r[i] = pFoot[i % 4][i / 4] - seResult.position[i / 4];

    double Q[12] = {15.0, 12.0, 10, 1.5, 1.5, 35, 0, 0, 0.3, 0.2, 0.2, 0.2};

    double yaw = seResult.rpy[2];
    double *weights = Q;
    double alpha = 4e-5; // make setting eventually

    // printf("current posistion: %3.f %.3f %.3f\n", p[0], p[1], p[2]);

    if (alpha > 1e-4)
    {
      std::cout << "Alpha was set too high (" << alpha << ") adjust to 1e-5\n";
      alpha = 1e-5;
    }
    Vec3<double> v_des_robot(stateCommand->data.stateDes[6], stateCommand->data.stateDes[7], 0);
    // Vec3<double> v_des_world = coordinateRotation(CoordinateAxis::Z, seResult.rpy[2]).transpose() * v_des_robot;

    Vec3<double> v_des_world = seResult.rBody.transpose() * v_des_robot;

    const double max_pos_error = .1;
    double xStart = world_position_desired[0];
    double yStart = world_position_desired[1];

    if (xStart - p[0] > max_pos_error)
      xStart = p[0] + max_pos_error;
    if (p[0] - xStart > max_pos_error)
      xStart = p[0] - max_pos_error;

    if (yStart - p[1] > max_pos_error)
      yStart = p[1] + max_pos_error;
    if (p[1] - yStart > max_pos_error)
      yStart = p[1] - max_pos_error;

    world_position_desired[0] = xStart;
    world_position_desired[1] = yStart;

    double trajInitial[12] = {                                              // stateCommand->data.stateDes[3],
                              rpy_comp[0] + stateCommand->data.stateDes[3], // 0 r
                              rpy_comp[1] + stateCommand->data.stateDes[4], // 1 p
                              stateCommand->data.stateDes[5],               // 2 y
                              xStart,                                       // 3 x
                              yStart,                                       // 4 y
                              0.32,                                         // 5 z !0714 this should set the z height, ori 0.3
                              0,                                            // 6 dr
                              0,                                            // 7 dp
                              stateCommand->data.stateDes[11],              // 8 dy
                              v_des_world[0],                               // 9 dx
                              v_des_world[1],                               // 10 dy
                              v_des_world[2]};                              // 11 dz
    if (climb)
    {
      trajInitial[1] = ground_pitch;
    }

    for (int i = 0; i < horizonLength; i++)
    {
      for (int j = 0; j < 12; j++)
        trajAll[12 * i + j] = trajInitial[j];
      if (i != 0)
      {
        trajAll[12 * i + 3] = trajAll[12 * (i - 1) + 3] + dtMPC * v_des_world[0];
        trajAll[12 * i + 4] = trajAll[12 * (i - 1) + 4] + dtMPC * v_des_world[1];
        trajAll[12 * i + 2] = trajAll[12 * (i - 1) + 2] + dtMPC * stateCommand->data.stateDes[11];
        // std::cout << "yaw traj" <<  trajAll[12*i + 2] << std::endl;
      }
    }

    Timer t1;
    t1.start();
    dtMPC = dt * iterationsBetweenMPC;
    Timer t2;
    t2.start();
    solver.update_problem_data(p, v, q, w, r, yaw, weights, trajAll, alpha, mpcTable, data._quadruped->robot_index);

    // t2.stopPrint("Run MPC");
    // printf("MPC Solve time %f ms\n", t2.getMs());
    // std::cout << t2.getSeconds() << std::endl;
    for (int leg = 0; leg < 4; leg++)
    {
      Vec3<double> f;
      for (int axis = 0; axis < 3; axis++)
        f[axis] = solver.get_solution(leg * 3 + axis);

      f_ff[leg] = -seResult.rBody * f;
    }
  }
}
