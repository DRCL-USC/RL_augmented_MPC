#ifndef FSM_H
#define FSM_H

#include "FSMState.h"
#include "FSMState_Passive.h"
#include "FSMState_PDStand.h"
#include "FSMState_QPStand.h"
#include "FSMState_Walking.h"
#include "FSMState_ThreeFoot.h"
#include "FSMState_rlMPC.h"
#include "../common/enumClass.h"
#include <fstream>

struct FSMStateList{
    FSMState *invalid;
    FSMState_Passive *passive;
    FSMState_PDStand *pdstand;
    FSMState_QPStand *qpstand;
    FSMState_Walking *walking;
    FSMState_ThreeFoot *threefoot;
    FSMState_RL *rltest;
   
    void deletePtr(){
        delete invalid;
        delete passive;
        delete qpstand;
        delete walking;
        delete threefoot;
        delete rltest;
    }  
};

class FSM{
    public:
        FSM(ControlFSMData *data);
        ~FSM();
        void initialize();
        void run();
    private:
        FSMState* getNextState(FSMStateName stateName);
        bool checkSafty();
        ControlFSMData *_data;
        FSMState *_currentState;
        FSMState *_nextState;
        FSMStateName _nextStateName;
        FSMStateList _stateList;
        FSMMode _mode;
        long long _startTime;
        int count;
        std::ofstream exp_data;
};

#endif