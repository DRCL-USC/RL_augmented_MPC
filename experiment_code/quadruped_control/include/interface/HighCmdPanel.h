#ifndef HIGHCMDPENAL_H
#define HIGHCMDPENAL_H

#include "../common/cppTypes.h"
#include "CmdPanel.h"
#include "../sdk/include/unitree_legged_sdk.h"

using namespace UNITREE_LEGGED_SDK;

class HighCmdPanel: public CmdPanel{
    public:
        HighCmdPanel();
        ~HighCmdPanel();
        
    private:
        HighCmd _highCmd = {0};
        HighState _highState = {0};
        UDP high_udp;
};

#endif