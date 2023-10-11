#ifndef FOOT_REACTION_POLICY
#define FOOT_REACTION_POLICY

#include "policyIO.h"

class footReactionPolicy: public PolicyInterface
{
    public:
        footReactionPolicy(int _num_obs, int _num_action, std::string _filename);
        ~footReactionPolicy(){}

        void update_action();
        void update_observation();

        Eigen::Matrix<double, 1, Dynamic> current_obs;
        Eigen::Matrix<double, 1, Dynamic> output_action;

        Eigen::Matrix<double, 1, Dynamic> action_ub;
        Eigen::Matrix<double, 1, Dynamic> action_lb;

        Eigen::Matrix<double, 1, 256> layer_1_out;
        Eigen::Matrix<double, 1, 32> layer_2_out;
        Eigen::Matrix<double, 1, 256> layer_3_out;
        Eigen::Matrix<double, 1, Dynamic> last_layer_out;

        Eigen::Matrix<double, 1, Dynamic> network_action_out;
};

#endif