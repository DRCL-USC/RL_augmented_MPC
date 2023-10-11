#include "../../include/FSM/FSMState_QPStand.h"

FSMState_QPStand::FSMState_QPStand(ControlFSMData *data)
                 :FSMState(data, FSMStateName::QPSTAND, "QPStand")
{
    _rollMax = 30 * M_PI / 180;
    _rollMin = -_rollMax;
    _pitchMax = 25 * M_PI / 180;
    _pitchMin = -_pitchMax;
    _yawMax = 30 * M_PI / 180;
    _yawMin = -_yawMax;
    _heightMax = 0.35;
    _heightMin = 0.15;
    _forwardMax = 0.1;
    _backwardMax = -0.1;
    
    footFeedForwardForces.setZero();
    for(int i = 0; i < 4; i++)
    {
        minForces[i] = minForce;
        maxForces[i] = maxForce;
    }
}

void FSMState_QPStand::enter()
{   
    counter = 0;
     _data->_interface->zeroCmdPanel();
    _data->_legController->zeroCommand();
    _data->_legController->updateData(_data->_lowState);
    _data->_stateEstimator->run();

    init_yaw = _data->_stateEstimator->getResult().rpy(2);
    if(_data->_quadruped->robot_index == 1){
        for(int i = 0; i < 3; i++)
        {
            p_des[i] = _data->_stateEstimator->getResult().position[i];
            rpy[i] = 0; 
            kpCOM[i] = 30;  
            kdCOM[i] = 10;  
            kpBase[i] = 100;  
            kdBase[i] = 20; 
        }
        
        kpCOM[2] = 95;
        kpBase[0] = 600;
        kpBase[1] = 400; 
        rpy[2] = init_yaw;
        p_des[2] = 0.43;
        balanceController.Ig << 0.050874, 0, 0, 0, 0.64036, 0, 0, 0, 0.65655; 
    }
    else if(_data->_quadruped->robot_index == 2){
        for(int i = 0; i < 3; i++)
        {
            p_des[i] = _data->_stateEstimator->getResult().position[i];
            rpy[i] = 0; 
            kpCOM[i] = 30;  
            kdCOM[i] = 10;  
            kpBase[i] = 200;  
            kdBase[i] = 20; 
        }
        
        kpCOM[2] = 105;
        kpBase[0] = 600;
        kpBase[1] = 500; 
        rpy[2] = init_yaw;
        p_des[2] = 0.3;
        balanceController.Ig << .0168, 0.0, 0.0, 0.0, 0.0565, 0.0, 0.0, 0.0, 0.064;
    }

    else if(_data->_quadruped->robot_index == 3){
        for(int i = 0; i < 3; i++)
        {
            p_des[i] = 0;
            rpy[i] = 0; 
            kpCOM[i] = 40;  
            kdCOM[i] = 15;  
            kpBase[i] = 80;  
            kdBase[i] = 30; 
        }
        
        kpCOM[2] = 50;
        kpBase[0] = 600;
        kpBase[1] = 500; 
        rpy[2] = init_yaw;
        p_des[2] = 0.4;
        balanceController.Ig << 0.1831 , 0, 0, 0, 0.7563, 0, 0, 0, 0.7837; 
    }
    else{
        std::cout << "robot not defined for QP controller" << std::endl;
        exit();
    }
}

template<typename T0, typename T1, typename T2>
T1 invNormalize(const T0 value, const T1 min, const T2 max, const double minLim = -1, const double maxLim = 1){
	return (value-minLim)*(max-min)/(maxLim-minLim) + min;
}


void FSMState_QPStand::run()
{
    _data->_legController->updateData(_data->_lowState);
    _data->_stateEstimator->run();
    _userValue = _data->_lowState->userValue;

    for (int i = 0; i < 4; i++) {
        se_xfb[i] = _data->_stateEstimator->getResult().orientation(i);
    }
     
    for (int i = 0; i < 3; i++) {
        p_act[i] = _data->_stateEstimator->getResult().position(i);
        
        v_act[i] = _data->_stateEstimator->getResult().vWorld(i);

        se_xfb[4 + i] = _data->_stateEstimator->getResult().position(i);
        se_xfb[7 + i] = _data->_stateEstimator->getResult().omegaWorld(i);
        se_xfb[10 + i] = _data->_stateEstimator->getResult().vWorld(i);
      }

          Vec3<double> pFeetVecCOM;

    // Get the foot locations relative to COM
    for (int leg = 0; leg < 4; leg++) {
        pFeetVecCOM =  _data->_stateEstimator->getResult().rBody.transpose() *
        (_data->_quadruped->getHipLocation(leg) + _data->_legController->data[leg].p);

        pFeet[leg * 3] = pFeetVecCOM[0];
        pFeet[leg * 3 + 1] = pFeetVecCOM[1];
        pFeet[leg * 3 + 2] = pFeetVecCOM[2];
        //std::cout << "pFeet" << leg << std::endl;
    }

    rpy[0] = (double)invNormalize(_userValue.lx, _rollMin, _rollMax);
    rpy[1] = (double)invNormalize(_userValue.ly, _pitchMin, _pitchMax);
    rpy[2] = init_yaw + (double)invNormalize(_userValue.rx, _yawMin, _yawMax);

    //p_des[0] = p_act[0] + (pFeet[0] + pFeet[3] + pFeet[6] + pFeet[9]) / 4.0;
    //p_des[1] = p_act[1] + (pFeet[1] + pFeet[4] + pFeet[7] + pFeet[10]) / 4.0;

    balanceController.set_alpha_control(0.01);
    balanceController.set_friction(0.4);
    balanceController.set_mass(_data->_quadruped->mass);
    balanceController.set_wrench_weights(COM_weights_stance, Base_weights_stance);
    balanceController.set_PDgains(kpCOM, kdCOM, kpBase, kdBase);
    balanceController.set_desiredTrajectoryData(rpy, p_des, omegaDes, v_des);
    balanceController.SetContactData(contactStateScheduled, minForces, maxForces);
    balanceController.updateProblemData(se_xfb, pFeet, p_des, p_act, v_des, v_act,
                                      O_err, _data->_stateEstimator->getResult().rpy(2));
   // balanceController.print_QPData();
    double fOpt[12];
    balanceController.solveQP_nonThreaded(fOpt);


    for (int leg = 0; leg < 4; leg++) {
        footFeedForwardForces.col(leg) << fOpt[leg * 3], fOpt[leg * 3 + 1],
        fOpt[leg * 3 + 2]; // force in world frame, need to convert to body frame
        
        _data->_legController->commands[leg].feedforwardForce = footFeedForwardForces.col(leg);
	    _data->_legController->commands[leg].kdJoint.diagonal() << 0.5, 0.5 ,0.5;
    }

    _data->_legController->updateCommand(_data->_lowCmd);
    counter++;
}

void FSMState_QPStand::exit()
{
    _data->_interface->zeroCmdPanel();
    _data->_interface->cmdPanel->setCmdNone();
}

FSMStateName FSMState_QPStand::checkTransition()
{
    if(_lowState->userCmd == UserCommand::L2_A){
        std::cout << "transition from QP stand to PD stand" << std::endl;
        return FSMStateName::PDSTAND;
    }
    else if(_lowState->userCmd == UserCommand::L2_B){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::L2_Y){
        std::cout << "transition from QP stand to 3 foot" << std::endl;
        return FSMStateName::THREEFOOT;
    }
    else if(_lowState->userCmd == UserCommand::START){
        std::cout << "transition from QP stand to walk" << std::endl;
        return FSMStateName::WALKING;
    }
    else if(_lowState->userCmd == UserCommand::L2_X){
        std::cout << "transition from QP stand to climb" << std::endl;
        return FSMStateName::CLIMB;
    }
    else{
        return FSMStateName::QPSTAND;
    }
}
