#ifndef THREEFOOT_H
#define THREEFOOT_H

#include "FSMState.h"
#include "../../BalanceController/BalanceController.hpp"

class FSMState_ThreeFoot: public FSMState
{
    public:
        FSMState_ThreeFoot(ControlFSMData *data);
        ~FSMState_ThreeFoot(){}
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();

        void setswingleg(int id);

    private:
        BalanceController balanceController;
        int counter;
        Mat34<double> footFeedForwardForces; 
        // QP Data
        double minForce = 5;
        double maxForce = 300; // change for different robot
        double minForces[4];
        double maxForces[4];
        double contactStateScheduled[4] = {1, 1, 1, 1}; //assume 3-leg standing
        double COM_weights_stance[3] = {5, 5, 10};
        double Base_weights_stance[3] = {10, 10, 20};
        double pFeet[12], p_act[3], v_act[3], O_err[3], rpy[3],v_des[3], p_des[3],
               omegaDes[3];
        double se_xfb[13];
        double kpCOM[3], kdCOM[3], kpBase[3], kdBase[3];
        double init_yaw;

        int swing_leg_id = 0;
        // Saturation
        double _rollMax, _rollMin;
        double _pitchMax, _pitchMin;
        double _yawMax, _yawMin;
        double _heightMax, _heightMin;
        double _forwardMax, _backwardMax;

};

#endif