#include "../../include/FSM/FSMState_Walking.h"

FSMState_Walking::FSMState_Walking(ControlFSMData *data)
                 :FSMState(data, FSMStateName::WALKING, "walking"),
                 Cmpc(0.001, 30, 10){}

template<typename T0, typename T1, typename T2>
T1 invNormalize(const T0 value, const T1 min, const T2 max, const double minLim = -1, const double maxLim = 1){
	return (value-minLim)*(max-min)/(maxLim-minLim) + min;
}

void FSMState_Walking::enter()
{
    // v_des_body << 0, 0, 0;
    pitch = 0;
    roll = 0;
    //  _data->_interface->zeroCmdPanel();
    counter = 0;
    Cmpc.firstRun = true;
    _data->_desiredStateCommand->firstRun = true;
    _data->_stateEstimator->run(); 
    // _data->_legController->zeroCommand();
}

void FSMState_Walking::run()
{
    _data->_legController->updateData(_data->_lowState);
    _data->_stateEstimator->run(); 
    _userValue = _data->_lowState->userValue;
    // adjusthte velocity range based on the gait cycle time
    v_des_body[0] = (double)invNormalize(_userValue.ly, -1.0, 1.0);
    v_des_body[1] = (double)invNormalize(_userValue.rx, 0.5, -0.5);
    turn_rate = (double)invNormalize(_userValue.lx, 2.0, -2.0);
    // roll = (double)invNormalize(_userValue.ry, -0.8, 0.8);

    if(walking2QP)
    {
        v_des_body[0] = 0; 
        v_des_body[1] = 0; 
        turn_rate = 0;
    }



    _data->_desiredStateCommand->setStateCommands(roll, pitch, v_des_body, turn_rate);
    
    // Cmpc.climb = true;
    Cmpc.setGaitNum(2); 
    Cmpc.run(*_data);

    _data->_legController->updateCommand(_data->_lowCmd);  
}

void FSMState_Walking::exit()
{      
    counter = 0; 
    _data->_interface->cmdPanel->setCmdNone();
}

FSMStateName FSMState_Walking::checkTransition()
{
    if(_lowState->userCmd == UserCommand::L1_X || walking2QP){
        if(Cmpc.phase > 0.96 && 
           abs(_data->_stateEstimator->getResult().vWorld(0)) < 0.2 && 
           abs(_data->_stateEstimator->getResult().vWorld(1)) < 0.2 &&
           abs(_data->_stateEstimator->getResult().omegaWorld(2) < 0.4)){
            std::cout << "transition from walking to QP stand" << std::endl;
            walking2QP = false;
            return FSMStateName::QPSTAND;
        }
        else{
            walking2QP = true;
            return FSMStateName::WALKING;
        }
    }
    else if(_lowState->userCmd == UserCommand::L2_X || walking2climb){
        if(Cmpc.phase > 0.96){
            std::cout << "transition from walking to stair climbing" << std::endl;
            walking2climb = false;
            return FSMStateName::CLIMB;
        }
        else{
            walking2climb = true;
            return FSMStateName::WALKING;
        }   
    }
    else if(_lowState->userCmd == UserCommand::L2_B){
        std::cout << "transition from Walking to passive" << std::endl;
        return FSMStateName::PASSIVE;
    }
    else{
        return FSMStateName::WALKING;
    }
}
