#include "../../include/FSM/FSMState_rlMPC.h"

FSMState_RL::FSMState_RL(ControlFSMData *data)
               :FSMState(data, FSMStateName::CLIMB, "climb"),
               Cmpc(0.001, 30, 10){}

template<typename T0, typename T1, typename T2>
T1 invNormalize(const T0 value, const T1 min, const T2 max, const double minLim = -1, const double maxLim = 1){
	return (value-minLim)*(max-min)/(maxLim-minLim) + min;
}

void FSMState_RL::enter()
{
    // v_des_body << 0, 0, 0;
    pitch = 0;
    roll = 0;
    // _data->_interface->zeroCmdPanel();
    counter = 0;
    _data->_desiredStateCommand->firstRun = true;
    Cmpc.firstRun = true;
    _data->_stateEstimator->run(); 
    // _data->_legController->zeroCommand();
}

void FSMState_RL::run()
{
    _data->_legController->updateData(_data->_lowState);
    _data->_stateEstimator->run(); 
    _userValue = _data->_lowState->userValue;
    // adjusthte velocity range based on the gait cycle time
    v_des_body[0] = (double)invNormalize(_userValue.ly, -0.5, 0.5);
    v_des_body[1] = (double)invNormalize(_userValue.rx, 0.5, -0.5);
    turn_rate = (double)invNormalize(_userValue.lx, 2.0, -2.0);

    _data->_desiredStateCommand->setStateCommands(roll, pitch, v_des_body, turn_rate);
    // Cmpc.climb = true;
    Cmpc.setGaitNum(2); 
    Cmpc.run(*_data);

    _data->_legController->updateCommand(_data->_lowCmd);  
}

void FSMState_RL::exit()
{   
    // Cmpc.firstRun = true;
    counter = 0; 
     for(int i = 0; i < 4; i++)
     {
	_data->_legController->q_offset[i].setZero();
	}
     _data->_legController->zeroCommand();
     _data->_interface->zeroCmdPanel();
    //_data->_legController->updateCommand(_data->_lowCmd);
    //_data->_interface->cmdPanel->setCmdNone();
}

FSMStateName FSMState_RL::checkTransition()
{
    if(_lowState->userCmd == UserCommand::START || climb2walking){
        if(Cmpc.phase > 0.95){
            climb2walking = false;
            return FSMStateName::WALKING;
        }
        else{
            climb2walking = true;
            return FSMStateName::CLIMB;
        }  
    }
    else if(_lowState->userCmd == UserCommand::L2_B){
        std::cout << "transition from PD stand to passive" << std::endl;
        return FSMStateName::PASSIVE;
    }
    else if (_lowState->userCmd == UserCommand::L1_X){
        return FSMStateName::QPSTAND;
    }

    else{
        return FSMStateName::CLIMB;
    }
}
