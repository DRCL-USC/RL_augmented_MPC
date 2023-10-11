#include "footReactionPolicy.h"

footReactionPolicy::footReactionPolicy(int _num_obs, int _num_action, std::string _filename):
    PolicyInterface(_num_obs, _num_action, _filename)
{
    current_obs.resize(Eigen::NoChange, _num_obs);
    output_action.resize(Eigen::NoChange, _num_action);
    action_ub.resize(Eigen::NoChange, _num_action);
    action_lb.resize(Eigen::NoChange, _num_action);

    last_layer_out.resize(Eigen::NoChange, _num_action*2);
    network_action_out.resize(Eigen::NoChange, _num_action);

    current_obs.setZero();
    output_action.setZero();
    action_ub.setZero();
    action_lb.setZero();
    network_action_out.setZero();
    output_action.setZero();

   action_ub << 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3;
   action_lb << -2.0, -2.0, -2.0, -1.0, -1.0, -2.0, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3;

}

void footReactionPolicy::update_observation()
{
    for(int i = 4; i > 0; i--)
    {
        Observation.block(0, num_obs * i, i, num_obs) = Observation.block(0, num_obs*(i-1), 1, num_obs);
    }

    Observation.block(0, 0, 1, num_obs) = current_obs;
    //std::cout << "observation " << Observation << std::endl;
}

void footReactionPolicy::update_action()
{   
    for(int i = 0; i < num_obs * 5; i++)
        {Normalized_obs(i) = (Observation(i) - obs_mean(i)) / sqrt(obs_variance(i) + 1e-10);}
   // std::cout << "obs shape " << Normalized_obs.rows() << " " << Normalized_obs.cols() << std::endl;
     //  std::cout << "layer1 shape " << layer_1_weights.rows() << " " << layer_1_weights.cols() << std::endl;
      // std::cout << "layer1 bias shape " << layer_1_bias.rows() << " " << layer_1_bias.cols() << std::endl;
      // std::cout << "layer2 shape " << layer_2_weights.rows() << " " << layer_2_weights.cols() << std::endl;
    //   std::cout << "layer2 bias shape " << layer_2_bias.rows() << " " << layer_2_bias.cols() << std::endl;
  //     std::cout << "layer3 shape " << layer_3_weights.rows() << " " << layer_3_weights.cols() << std::endl;
//       std::cout << "layer4 bias shape " << layer_4_bias.rows() << " " << layer_4_bias.cols() << std::endl;
    //std::cout << "normalized obs " << Normalized_obs << std::endl;
    layer_1_out = (Normalized_obs * layer_1_weights + layer_1_bias).array().tanh();
    layer_2_out = (layer_1_out * layer_2_weights + layer_2_bias).array().tanh();
    layer_3_out = (layer_2_out * layer_3_weights + layer_3_bias).array().tanh();
    last_layer_out = (layer_3_out * layer_4_weights + layer_4_bias).array();
  //  std::cout << "compute " << std::endl;
    network_action_out = last_layer_out.block(0, 0, 1, num_action);
   // std::cout  << network_action_out << std::endl;
    for(int i = 0; i < num_action; i++)
    {
        if(network_action_out(i) > 1.0)
            network_action_out(i) = 1.0;
        
        if(network_action_out(i) < -1.0)
            network_action_out(i) = -1.0;
        
        output_action(i) = action_lb(i) + 0.5 * (network_action_out(i) + 1.0) * (action_ub(i) - action_lb(i));
    }
//    std::cout << " final output" << output_action << std::endl;
}   
