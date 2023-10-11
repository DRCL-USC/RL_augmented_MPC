#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include "../messages/LowLevelCmd.h"
#include "../messages/LowlevelState.h"
#include "CmdPanel.h"
#include <string>

class IOInterface
{
    public:
        IOInterface(){}
        ~IOInterface(){}
        virtual void sendRecv(const LowlevelCmd *cmd, LowlevelState *state) = 0;
        void zeroCmdPanel(){cmdPanel->setZero();}
        void setPassive(){cmdPanel->setPassive();}
        CmdPanel *cmdPanel;
};

#endif