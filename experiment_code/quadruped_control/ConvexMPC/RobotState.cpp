#include "RobotState.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>

using std::cout;
using std::endl;

void RobotState::set(flt* p_, flt* v_, flt* q_, flt* w_, flt* r_,flt yaw_, int robot_type)
{
    for(u8 i = 0; i < 3; i++)
    {
        this->p(i) = p_[i];
        this->v(i) = v_[i];
        this->w(i) = w_[i];
    }
    this->q.w() = q_[0];
    this->q.x() = q_[1];
    this->q.y() = q_[2];
    this->q.z() = q_[3];
    this->yaw = yaw_;

    //for(u8 i = 0; i < 12; i++)
    //    this->r_feet(i) = r[i];
    for(u8 rs = 0; rs < 3; rs++)
        for(u8 c = 0; c < 4; c++)
            this->r_feet(rs,c) = r_[rs*4 + c];

    R = this->q.toRotationMatrix();
    fpt yc = cos(yaw_);
    fpt ys = sin(yaw_);

    R_yaw <<  yc,  -ys,   0,
             ys,  yc,   0,
               0,   0,   1;

    Eigen::Matrix<fpt,3,1> Id;
    if (robot_type == 2)
    {
        // Id << .0168f, 0.0565f, 0.064f; // A1
        Id << 0.0302f,  0.2507f, 0.273f;
        // Id << 0.232, 0.4075, 0.3775;
        m = 12.0;
    }
    else if(robot_type == 1)
    {
        // Id << 0.050874f, 0.64036f, 0.65655f; // Aliengo
	    Id << 0.0875, 1.293, 1.338;
        m = 20.0;
    }

    I_body.diagonal() = Id;
    I_world = R_yaw * I_body * R_yaw.transpose(); 
}

void RobotState::print()
{
   cout<<"Robot State:"<<endl<<"Position\n"<<p.transpose()
       <<"\nVelocity\n"<<v.transpose()<<"\nAngular Veloctiy\n"
       <<w.transpose()<<"\nRotation\n"<<R<<"\nYaw Rotation\n"
       <<R_yaw<<"\nFoot Locations\n"<<r_feet<<"\nInertia\n"<<I_body<<endl;
}



