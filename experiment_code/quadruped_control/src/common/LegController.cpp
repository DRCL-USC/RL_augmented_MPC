#include "../../include/common/LegController.h"
#include <iostream>

// upper level of joint controller 
// send data to joint controller

void LegControllerCommand::zero(){
    tau = Vec3<double>::Zero();
    qDes = Vec3<double>::Zero();
    qdDes = Vec3<double>::Zero();
    pDes = Vec3<double>::Zero();
    vDes = Vec3<double>::Zero();
    feedforwardForce = Vec3<double>::Zero();
    kpCartesian = Mat3<double>::Zero(); 
    kdCartesian = Mat3<double>::Zero();
    kpJoint = Mat3<double>::Zero();
    kdJoint = Mat3<double>::Zero();
}

/*!
 * Zero leg data
 */ 
void LegControllerData::zero(){
    q = Vec3<double>::Zero();
    qd = Vec3<double>::Zero();
    p = Vec3<double>::Zero();
    v = Vec3<double>::Zero();
    J = Mat3<double>::Zero();
    tau = Vec3<double>::Zero();
}

void LegController::zeroCommand(){
    for (int i = 0; i<4; i++){
        commands[i].zero();
        q_offset[i].setZero();
    }
}

void LegController::updateData(const LowlevelState* state){
    for (int leg = 0; leg < 4; leg++){
        for(int j = 0; j<3; j++){
            data[leg].q(j) = state->motorState[leg*3+j].q;
            data[leg].qd(j) = state->motorState[leg*3+j].dq;
            data[leg].tau(j) = state->motorState[leg*3+j].tauEst;
        }

        computeLegJacobianAndPosition(_quadruped, data[leg].q,&(data[leg].J),&(data[leg].p),leg);

         // v
        data[leg].v = data[leg].J * data[leg].qd;
    }
}

void LegController::updateCommand(LowlevelCmd* cmd){

    for (int i = 0; i <4; i++){
        Vec3<double> legTorque = commands[i].tau;
        // forceFF
        Vec3<double> footForce = commands[i].feedforwardForce;

        if(commands[i].pDes[2] != 0){ // make sure it doesn't work when foot pos command is not used
            computeInverseKinematics(_quadruped, commands[i].pDes, i, &(commands[i].qDes));
            commands[i].qdDes = data[i].J.transpose() * commands[i].vDes;
        }

        // torque
        legTorque += data[i].J.transpose() * footForce;

        commands[i].tau += legTorque;
        for (int j = 0; j < 3; j++){
            cmd->motorCmd[i*3+j].tau = commands[i].tau(j);
            if(!commands[i].qDes.hasNaN())
                cmd->motorCmd[i*3+j].q = commands[i].qDes(j) + q_offset[i][j];
            cmd->motorCmd[i*3+j].dq = commands[i].qdDes(j);
            cmd->motorCmd[i*3+j].Kp = commands[i].kpJoint(j,j);
            cmd->motorCmd[i*3+j].Kd = commands[i].kdJoint(j,j);
        }
        commands[i].tau << 0, 0, 0; // zero torque command to prevent interference
    }
    //std::cout << "cmd sent" << std::endl;
   
}

void computeLegJacobianAndPosition(Quadruped& _quad, Vec3<double>& q, Mat3<double>* J,Vec3<double>* p, int leg)
{
    double l1 = _quad.hipLinkLength; // ab_ad
    double l2 = _quad.thighLinkLength;
    double l3 = _quad.calfLinkLength;

    int sideSign = 1; // 1 for Left legs; -1 for right legs
    if (leg == 0 || leg == 2){
        sideSign = -1;
    }

    double s1 = std::sin(q(0));
    double s2 = std::sin(q(1));
    double s3 = std::sin(q(2));

    double c1 = std::cos(q(0));
    double c2 = std::cos(q(1));
    double c3 = std::cos(q(2));

    double c23 =  c2 * c3 - s2 * s3;
    double s23 =  s2 * c3 + c2 * s3; // sin(2+3))
   
   if(J){
    J->operator()(0, 0) = 0;
    J->operator()(1, 0) = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1;
    J->operator()(2, 0) = sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1;
    J->operator()(0, 1) = -l3 * c23 - l2 * c2;
    J->operator()(1, 1) = -l2 * s2 * s1 - l3 * s23 * s1;
    J->operator()(2, 1) = l2 * s2 * c1 + l3 * s23 * c1;
    J->operator()(0, 2) = -l3 * c23;
    J->operator()(1, 2) = -l3 * s23 *s1;
    J->operator()(2, 2) = l3 * s23 * c1;   
   }

   if(p){
    p->operator()(0) = -l3 * s23 - l2 * s2;
    p->operator()(1) = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1;
    p->operator()(2) = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2;
   }
}

double q1_ik(double py, double pz, double l1)
{
    double q1;
    double L = sqrt(pow(py,2)+pow(pz,2)-pow(l1,2));
    q1 = atan2(pz*l1+py*L, py*l1-pz*L);

    return q1;
}

double q2_ik(double q1, double q3, double px, double py, double pz, double b3z, double b4z){
    double q2, a1, a2, m1, m2;
    
    a1 = py*sin(q1) - pz*cos(q1);
    a2 = px;
    m1 = b4z*sin(q3);
    m2 = b3z + b4z*cos(q3);
    q2 = atan2(m1*a1+m2*a2, m1*a2-m2*a1);
    return q2;
}

double q3_ik(double b3z, double b4z, double b){
    double q3, temp;
    temp = (pow(b3z, 2) + pow(b4z, 2) - pow(b, 2))/(2*fabs(b3z*b4z));
    if(temp>1) temp = 1;
    if(temp<-1) temp = -1;
    q3 = acos(temp);
    q3 = -(M_PI - q3); //0~180
    return q3;
}

void computeInverseKinematics(Quadruped& _quad, Vec3<double>& pDes, int leg, Vec3<double>* qDes)
{
    double l1 = _quad.hipLinkLength; // ab_ad
    double l2 = _quad.thighLinkLength;
    double l3 = _quad.calfLinkLength;
    
    double q1, q2, q3;
    double b2y, b3z, b4z, a, b, c;
    int sideSign = 1; // 1 for Left legs; -1 for right legs
    if (leg == 0 || leg == 2){
        sideSign = -1;
    }

    b2y = l1 * sideSign;
    b3z = -l2;
    b4z = -l3;
    a = l1;
    c = sqrt(pow(pDes(0), 2) + pow(pDes(1), 2) + pow(pDes(2), 2)); // whole length
    b = sqrt(pow(c, 2) - pow(a, 2)); // distance between shoulder and footpoint

    q1 = q1_ik(pDes(1), pDes(2), b2y);
    q3 = q3_ik(b3z, b4z, b);
    q2 = q2_ik(q1, q3, pDes(0), pDes(1), pDes(2), b3z, b4z);

    qDes->operator()(0) = q1;
    qDes->operator()(1) = q2;
    qDes->operator()(2) = q3;
}

