#ifndef CMDPANEL_H
#define CMDPANEL_H

#include "../messages/unitree_joystick.h"
#include "../common/enumClass.h"
#include "../sdk/include/unitree_legged_sdk.h"
#include "../messages/LowlevelState.h"
#include <pthread.h>


class CmdPanel{
public:
    CmdPanel(){}
    ~CmdPanel(){}
    UserCommand getUserCmd(){return userCmd;}
    UserValue getUserValue(){return userValue;}
    void setPassive(){userCmd = UserCommand::L2_B;}
    void setZero(){userValue.setZero();}
    void setCmdNone(){userCmd = UserCommand::NONE;}
    virtual void receiveHandle(UNITREE_LEGGED_SDK::LowState *lowState){};
    virtual void setHighlevelMsg(UNITREE_LEGGED_SDK::HighState *highState){};
protected:
    virtual void *run(void *arg){};
    UserCommand userCmd;
    UserValue userValue;
};

#endif  // CMDPANEL_H
