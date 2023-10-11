#include "adaptivePolicy.h"


adaptivePolicy::adaptivePolicy(int _num_obs, int _num_action, std::string _filename):
    PolicyInterface(_num_obs, _num_action, _filename)
{
    // current_obs.resize(Eigen::NoChange, _num_obs);
    // output_action.resize(Eigen::NoChange, _num_action);
    // action_ub.resize(Eigen::NoChange, _num_action);
    // action_lb.resize(Eigen::NoChange, _num_action);

    // last_layer_out.resize(Eigen::NoChange, _num_action*2);
    // network_action_out.resize(Eigen::NoChange, _num_action);

    current_obs.setZero();
    output_action.setZero();
    action_ub.setZero();
    action_lb.setZero();
    network_action_out.setZero();
    output_action.setZero();

   action_ub << 5.0, 10.0, 2.0, 
                1.0, 1.0, 2.0, 
                0.1, 0.05, 
                0.1, 0.05, 
                0.05, 0.05, 
                0.05, 0.05;
 
   action_lb << -5.0, -10.0, -2.0, 
                -1.0, -1.0, -10.0, 
                -0.05, -0.05, 
                -0.05, -0.05, 
                -0.05, -0., 
                -0.05, -0.1;

    // std::cout << action_ub << "up \n" << action_lb  << "low \n ";

}

void adaptivePolicy::update_observation()
{
    for(int i = 4; i > 0; i--)
    {
        Observation.block(0, num_obs * i, i, num_obs) = Observation.block(0, num_obs*(i-1), 1, num_obs);
    }

    Observation.block(0, 0, 1, num_obs) = current_obs;
    //std::cout << "observation " << Observation << std::endl;
}

void adaptivePolicy::update_action()
{   
    for(int i = 0; i < num_obs * 5; i++)
        {Normalized_obs(i) = (Observation(i) - obs_mean(i)) / sqrt(obs_variance(i) + 1e-12);}

    layer_1_out = (Normalized_obs * layer_1_weights + layer_1_bias).array().tanh();
    layer_2_out = (layer_1_out * layer_2_weights + layer_2_bias).array().tanh();
    layer_3_out = (layer_2_out * layer_3_weights + layer_3_bias).array().tanh();
    last_layer_out = (layer_3_out * layer_4_weights + layer_4_bias).array();
  //  std::cout << "compute " << std::endl;
    network_action_out = last_layer_out.block(0, 0, 1, 14);
   // std::cout  << network_action_out << std::endl;
    for(int i = 0; i < num_action; i++)
    {
        if(network_action_out(i) > 1.0)
            network_action_out(i) = 1.0;
        
        if(network_action_out(i) < -1.0)
            network_action_out(i) = -1.0;
        
        output_action(i) = action_lb(i) + 0.5 * (network_action_out(i) + 1.0) * (action_ub(i) - action_lb(i));
    }
}   
