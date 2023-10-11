
#include "../../include/interface/IOSDK.h"
#include "../../include/interface/WirelessHandle.h"
#include "../../include/interface/KeyBoard.h"
#include "../../include/sdk/include/unitree_legged_sdk.h"
#include <stdio.h>

using namespace UNITREE_LEGGED_SDK;

IOSDK::IOSDK(LeggedType robot, int cmd_panel_id):
_control(robot),
_udp(LOWLEVEL),
_udp_low_cmd(8002, "127.0.0.1", 8001, sizeof(LowCmd), sizeof(LowCmd)),
_udp_low_state(8083, "127.0.0.1", 8084,  sizeof(LowState), sizeof(LowState)),
_udp_high_state(8082, "127.0.0.1", 8081, sizeof(HighState), sizeof(HighState))
{
    std::cout << "The control interface for real robot" << std::endl;
    _udp.InitCmdData(_lowCmd);
    if(cmd_panel_id == 1){
    cmdPanel = new WirelessHandle();
    }
    else if(cmd_panel_id == 2){
    cmdPanel = new KeyBoard();
    }
}

void IOSDK::sendRecv(const LowlevelCmd *cmd, LowlevelState *state){
    _udp.Recv();
    _udp.GetRecv(_lowState);
    for(int i(0); i < 12; ++i){
        _lowCmd.motorCmd[i].mode = 0X0A; 
        _lowCmd.motorCmd[i].q    = cmd->motorCmd[i].q;
        _lowCmd.motorCmd[i].dq   = cmd->motorCmd[i].dq;
        _lowCmd.motorCmd[i].Kp   = cmd->motorCmd[i].Kp;
        _lowCmd.motorCmd[i].Kd   = cmd->motorCmd[i].Kd;
        _lowCmd.motorCmd[i].tau  = cmd->motorCmd[i].tau;
    }


    for(int i(0); i < 12; ++i){
        state->motorState[i].q = _lowState.motorState[i].q;
        state->motorState[i].dq = _lowState.motorState[i].dq;
        state->motorState[i].tauEst = _lowState.motorState[i].tauEst;
        state->motorState[i].mode = _lowState.motorState[i].mode;
    }
    
    for(int i(0); i < 3; ++i){
        state->imu.quaternion[i] = _lowState.imu.quaternion[i];
        state->imu.gyroscope[i]  = _lowState.imu.gyroscope[i];
        state->imu.accelerometer[i] = _lowState.imu.accelerometer[i];
    }
    state->imu.quaternion[3] = _lowState.imu.quaternion[3];

    for(int i = 0; i < 4; i++){
        state->FootForce[i] = _lowState.footForce[i];
    }

    for(int i = 0; i < 3; i++)
    {
       _highState.position[i] = state->position[i];
       _highState.velocity[i] = state->vWorld[i];
    }

    _highState.imu = _lowState.imu;
    
    cmdPanel->receiveHandle(&_lowState);
    state->userCmd = cmdPanel->getUserCmd();
    state->userValue = cmdPanel->getUserValue();

    // _control.PowerProtect(_lowCmd, _lowState, 10);
    _udp_low_state.SetSend((char*) &_lowState);
    _udp_low_state.Send();

    _udp_low_cmd.SetSend((char*) &_lowCmd);
    _udp_low_cmd.Send();

    _udp_high_state.SetSend((char*) &_highState);
    _udp_high_state.Send();

    _udp.SetSend(_lowCmd);
    _udp.Send();
}