#include <iostream>
#include <unistd.h>
#include <csignal>
#include <sched.h>

#include "../include/interface/IOSDK.h"
#include "../include/common/ControlFSMData.h"
#include "../include/common/ContactEstimator.h"
#include "../include/common/OrientationEstimator.h"
#include "../include/common/PositionVelocityEstimator.h"
#include "../include/interface/WirelessHandle.h"
#include "../include/FSM/FSM.h"

using namespace UNITREE_LEGGED_SDK;

bool running = true;

void ShutDown(int sig)
{
    std::cout << "stop" << std::endl;
    running = false;
}

void setProcessScheduler()
{
    pid_t pid = getpid();
    sched_param param;
        param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if(sched_setscheduler(pid, SCHED_FIFO, &param) == -1)
    {
      std::cout << "[ERROR] function setprocessscheduler failed \n ";  
    }
}

int main()
{
    setProcessScheduler();
    
    double dt = 0.001;    
    int robot_id = 2; // AlienGo=1, A1=2
    int cmd_panel_id = 1; // Wireless=1, keyboard=2
    
    IOInterface *ioInter;
    if(robot_id == 1){
    ioInter = new IOSDK(LeggedType::Aliengo, cmd_panel_id);
    }
    else if(robot_id == 2){
    ioInter = new IOSDK(LeggedType::A1, cmd_panel_id);
    }
    Quadruped quad;
    quad.setQuadruped(robot_id);
    LegController* legController = new LegController(quad);
    LowlevelCmd* lowCmd = new LowlevelCmd();
    LowlevelState* lowState = new LowlevelState();
    StateEstimate stateEstimate;
    StateEstimatorContainer* stateEstimator = new StateEstimatorContainer(lowState,
                                                                          legController->data,
                                                                          &stateEstimate);
                                                                        
    stateEstimator->addEstimator<ContactEstimator>();
    stateEstimator->addEstimator<VectorNavOrientationEstimator>();
    stateEstimator->addEstimator<LinearKFPositionVelocityEstimator>();

    DesiredStateCommand* desiredStateCommand = new DesiredStateCommand(&stateEstimate, dt);

    ControlFSMData* _controlData = new ControlFSMData;
    _controlData->_quadruped = &quad;
    _controlData->_stateEstimator = stateEstimator;
    _controlData->_legController = legController;
    _controlData->_desiredStateCommand = desiredStateCommand;
    _controlData->_interface = ioInter;
    _controlData->_lowCmd = lowCmd;
    _controlData->_lowState = lowState;

    FSM* _FSMController = new FSM(_controlData);

    LoopFunc loop_control("control_loop", dt, boost::bind(&FSM::run, _FSMController));
    LoopFunc loop_udpSend("udp_send",     dt, 3, boost::bind(&ControlFSMData::sendRecv, _controlData));

    loop_udpSend.start();
    loop_control.start();


    while(1){
        +sleep(10);
    }
    
    delete _FSMController;
    delete _controlData;

    return 0;

}
