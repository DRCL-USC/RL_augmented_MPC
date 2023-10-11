#include "../../include/interface/WirelessHandle.h"
#include <string.h>
#include <stdio.h>

WirelessHandle::WirelessHandle(){}

void WirelessHandle::receiveHandle(UNITREE_LEGGED_SDK::LowState *lowState){
    memcpy(&_keyData, lowState->wirelessRemote, 40);

    if(((int)_keyData.btn.components.L2 == 1) && 
       ((int)_keyData.btn.components.B  == 1)){
        userCmd = UserCommand::L2_B;
    }
    else if(((int)_keyData.btn.components.L2 == 1) && 
            ((int)_keyData.btn.components.A  == 1)){
        userCmd = UserCommand::L2_A;
    }
    else if(((int)_keyData.btn.components.L2 == 1) && 
            ((int)_keyData.btn.components.X  == 1)){
        userCmd = UserCommand::L2_X;
    }

    else if(((int)_keyData.btn.components.L2 == 1) && 
            ((int)_keyData.btn.components.Y  == 1)){
        userCmd = UserCommand::L2_Y;
    }

    else if(((int)_keyData.btn.components.L1 == 1) && 
            ((int)_keyData.btn.components.X  == 1)){
        userCmd = UserCommand::L1_X;
    }
    else if(((int)_keyData.btn.components.L1 == 1) && 
            ((int)_keyData.btn.components.A  == 1)){
        userCmd = UserCommand::L1_A;
    }
    else if(((int)_keyData.btn.components.L1 == 1) && 
            ((int)_keyData.btn.components.Y  == 1)){
        userCmd = UserCommand::L1_Y;
    }
    else if((int)_keyData.btn.components.start == 1){
        userCmd = UserCommand::START;
    }


    userValue.L2 = _keyData.L2;
    double last_lx = userValue.lx;
    userValue.lx = 0.995*last_lx + 0.005*_keyData.lx;

    double last_ly = userValue.ly;
    userValue.ly = 0.995 * last_ly + 0.005 * _keyData.ly;
  
    double last_rx = userValue.rx;
    userValue.rx = 0.995*last_rx + 0.005*_keyData.rx;

    double last_ry = userValue.ry;
    userValue.ry =  0.995*last_ry + 0.005*_keyData.ry;

}
