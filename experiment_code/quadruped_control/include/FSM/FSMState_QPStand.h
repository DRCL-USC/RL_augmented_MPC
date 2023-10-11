#ifndef QPSTAND_H
#define QPSTAND_H

#include "FSMState.h"
#include "../../BalanceController/BalanceController.hpp"

class FSMState_QPStand: public FSMState
{
    public:
        FSMState_QPStand(ControlFSMData *data);
        ~FSMState_QPStand(){}
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();

    private:
        BalanceController balanceController;
        int counter;
        Mat34<double> footFeedForwardForces; 
        // QP Data
        double minForce = 5;
        double maxForce = 450; // change for different robot
        double minForces[4];
        double maxForces[4];
        double contactStateScheduled[4] = {1, 1, 1, 1}; //assume 4-leg standing
        double COM_weights_stance[3] = {5, 5, 5};
        double Base_weights_stance[3] = {40, 20, 10};
        double pFeet[12], p_act[3], v_act[3], O_err[3], rpy[3],v_des[3], p_des[3],
               omegaDes[3];
        double se_xfb[13];
        double kpCOM[3], kdCOM[3], kpBase[3], kdBase[3];
        double init_yaw;

        // Saturation
        double _rollMax, _rollMin;
        double _pitchMax, _pitchMin;
        double _yawMax, _yawMin;
        double _heightMax, _heightMin;
        double _forwardMax, _backwardMax;

};

#endif