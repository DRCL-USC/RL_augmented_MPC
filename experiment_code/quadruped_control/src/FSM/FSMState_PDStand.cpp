#include <iostream>
#include "../../include/FSM/FSMState_PDStand.h"

FSMState_PDStand::FSMState_PDStand(ControlFSMData *data)
                :FSMState(data, FSMStateName::PDSTAND, "PDStand"){}

void FSMState_PDStand::enter()
{
    _data->_legController->updateData(_data->_lowState);
    _data->_legController->zeroCommand();
    for(int i = 0; i < 4; i++)
    {
        _data->_legController->commands[i].kpJoint << 80, 0, 0,
                                                      0, 80, 0,
                                                      0, 0, 80;

        _data->_legController->commands[i].kdJoint << 3, 0, 0,
                                                      0, 3, 0,
                                                      0, 0, 3;
        for(int j = 0; j < 3; j++)
        {
            _startPos[i*3+j] = _data->_legController->data[i].q(j);
        }
    }
}

void FSMState_PDStand::run()
{
    _data->_legController->updateData(_data->_lowState);
    _data->_stateEstimator->run(); 
    _percent += 1.0/_duration;
    _percent = _percent > 1 ? 1 : _percent;
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            _data->_legController->commands[i].qDes(j) = (1 - _percent) * _startPos[i*3+j] + _percent * _targetPos[i*3+j];
        }

    }
    _data->_legController->updateCommand(_data->_lowCmd);
}

void FSMState_PDStand::exit()
{
    _percent = 0;
    _data->_interface->cmdPanel->setCmdNone();
     _data->_legController->zeroCommand();
}

FSMStateName FSMState_PDStand::checkTransition()
{
    if(_lowState->userCmd == UserCommand::L2_B){
        std::cout << "transition from PD stand to passive" << std::endl;
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L1_X){
        std::cout << "transition from PD stand to QP stand" << std::endl;
        return FSMStateName::QPSTAND;
    }
    else{
        return FSMStateName::PDSTAND;
    }
}
