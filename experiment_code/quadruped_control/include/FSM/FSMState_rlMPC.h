#ifndef RLMPC_H
#define RLMPC_H

#include "FSMState.h"
#include "../../ConvexMPC/rlMPC.h"
// #include "../../ConvexMPC/adaptiveMPC.h"

class FSMState_RL: public FSMState
{
    public:
        FSMState_RL(ControlFSMData *data);
        ~FSMState_RL(){}
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();
    
    private:
        rlMPC Cmpc;
        int counter;
        Vec3<double> v_des_body;
        double turn_rate = 0;
        double pitch, roll;
        bool climb2walking = false;
};

#endif