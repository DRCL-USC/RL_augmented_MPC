#ifndef WALKING_H
#define WALKING_H

#include "FSMState.h"
#include "../../ConvexMPC/adaptiveMPC.h"

class FSMState_Walking: public FSMState
{
    public:
        FSMState_Walking(ControlFSMData *data);
        ~FSMState_Walking(){}
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();
    
    private:
        adaptiveMPC Cmpc;
        int counter;
        Vec3<double> v_des_body;
        double turn_rate = 0;
        double pitch, roll;
        bool walking2climb = false;
        bool walking2QP = false;
};

#endif