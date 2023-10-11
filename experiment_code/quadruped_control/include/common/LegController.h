/*!
 * @file LegController.h
 * @brief Comman Leg Control Interface
 * 
 * Implement low-level leg control for Aliengo Robot
 * Leg 0: FR; Leg 1: FL;
 * Leg 2: RR ; Leg 3: RL;
 */ 

#ifndef LEGCONTROLLER_H
#define LEGCONTROLLER_H

#include "cppTypes.h"
#include "../messages/LowlevelState.h"
#include "../messages/LowLevelCmd.h"
#include "Quadruped.h"

/*!
 * Data sent from control algorithm to legs
 */ 
    struct LegControllerCommand{
        LegControllerCommand() {zero();}

        void zero();

        Vec3<double> qDes, qdDes, tau, pDes, vDes;
        Mat3<double> kpJoint, kdJoint;
        Vec3<double> feedforwardForce;
        Mat3<double> kpCartesian;
        Mat3<double> kdCartesian;
    };

/*!
 * Data returned from legs to control code
 */ 
    struct LegControllerData{
        LegControllerData() {zero();}
        void setQuadruped(Quadruped& quad) { aliengo = &quad; }

        void zero();
        Vec3<double> q, qd;
        Vec3<double> p, v;
        Mat3<double> J;
        Vec3<double> tau;
        Quadruped* aliengo;
    };

/*!
 * Controller for 4 legs of quadruped
 */ 
    class LegController {
      public:
        LegController(Quadruped& quad) : _quadruped(quad) {
            for (auto& dat : data) dat.setQuadruped(_quadruped);
            for(int i = 0; i<4; i++){
                commands[i].zero();
                data[i].zero();
                q_offset[i].setZero();
            }
        };
        
        void zeroCommand();
        void edampCommand(double gain);
        void updateData(const LowlevelState* state);
        void updateCommand(LowlevelCmd* cmd);

        Vec3<double> q_offset[4];
        LegControllerCommand commands[4];
        LegControllerData data[4];


        Quadruped& _quadruped;
        //CurrentState& curr;
        //ros::NodeHandle n;
    };

    void computeLegJacobianAndPosition(Quadruped& _quad, Vec3<double>& q, Mat3<double>* J,
                                       Vec3<double>* p, int leg);

    void computeInverseKinematics(Quadruped& _quad, Vec3<double>& pDes, int leg, Vec3<double>* qDes);

#endif