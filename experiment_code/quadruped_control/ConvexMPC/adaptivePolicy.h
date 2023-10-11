#ifndef SDAPTIVE_POLICY
#define ADAPTIVE_POLICY

#include "policyIO.h"

class adaptivePolicy: public PolicyInterface
{
    public:
        adaptivePolicy(int _num_obs, int _num_action, std::string _filename);
        ~adaptivePolicy(){}

        void update_action();
        void update_observation();

        Eigen::Matrix<double, 1, 78> current_obs;
        Eigen::Matrix<double, 1, 14> output_action;

        Eigen::Matrix<double, 1, 14> action_ub;
        Eigen::Matrix<double, 1, 14> action_lb;

        Eigen::Matrix<double, 1, 256> layer_1_out;
        Eigen::Matrix<double, 1, 32> layer_2_out;
        Eigen::Matrix<double, 1, 256> layer_3_out;
        Eigen::Matrix<double, 1, 28> last_layer_out;

        Eigen::Matrix<double, 1, 14> network_action_out;
};

#endif