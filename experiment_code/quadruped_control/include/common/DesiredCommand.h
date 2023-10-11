/*!
 * @file DesiredCommand.h
 * @brief convert command into desired
 * tracjectory for the robot
 */ 

#ifndef DESIREDCOMMAND_H
#define DESIREDCOMMAND_H

#include "cppTypes.h"
#include <signal.h>
#include <stdio.h>
#include <termios.h>
#include "StateEstimatorContainer.h"
#include "../interface/CmdPanel.h"

struct DesiredStateData{

    DesiredStateData() { zero(); }

    // Zero all data
    void zero();

    // Instataneous desired state comman
    Vec12<double> stateDes;
    Vec12<double> pre_stateDes;

    int mode;
};

class DesiredStateCommand {
  public:
    // Initialize
    DesiredStateCommand(StateEstimate* _estimate, double _dt){
      stateEstimate = _estimate;
      dt = _dt;
    }
    void setStateCommands(double r, double p, Vec3<double> v_des, double yaw_rate);
    void setmode(int ctrl_mode) {data.mode = ctrl_mode;}
    double deadband(double command, double minVal, double maxVal);
    // These should come from the inferface

    bool firstRun = true;

    DesiredStateData data;

    //~DesiredStateCommand();
  private:
    StateEstimate* stateEstimate;

    double dt; // Control loop time step
};



#endif