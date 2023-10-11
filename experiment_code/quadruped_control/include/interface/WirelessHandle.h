#ifndef WIRELESSHANDLE_H
#define WIRELESSHANDLE_H

#include "../messages/unitree_joystick.h"
#include "CmdPanel.h"
#include "../sdk/include/comm.h"

class WirelessHandle : public CmdPanel{
public:
    WirelessHandle();
    ~WirelessHandle(){}
    void receiveHandle(UNITREE_LEGGED_SDK::LowState *lowState);
private:
    xRockerBtnDataStruct _keyData;
    // LPFilter *_L2Value, *_lxValue, *_lyValue, *_rxValue, *_ryValue;
};

#endif  // WIRELESSHANDLE_H