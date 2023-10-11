#include "policyIO.h"

PolicyInterface::PolicyInterface(int _num_obs, int _num_action, std::string _filename):
num_obs(_num_obs),
num_action(_num_action)
{
    read_file(_filename);
    Observation.setZero();
    Normalized_obs.setZero();

    std::cout << " file read " << std::endl;
    // action_ub << 5.0, 10.0, 2.0, 2.0, 1.0, 6.0,
    //              0.05, 0.05,0.05, 0.05, 0.05, 0.05 ,0.05, 0.05;
    // action_lb << -5.0, -10.0, -2.0, -2.0, -1.0, 0.0, 
    //              -0.05, -0.05, -0.05, -0.05, -0.05, -0.05 ,-0.05, -0.05;
    // obs.open("obs.txt");
    // action_file.open("action.txt");
}

// void PolicyInterface::updateObservation(Eigen::Matrix<double, 1, 66> current_obs)
// {
//     // move current observations to the right 
//     for(int i = 4; i > 0; i--)
//     {
//         Observation.block(0, 66*i, 1, 66) = Observation.block(0, 66*(i-1), 1, 66);
//     }
//     Observation.block(0,0,1,66) = current_obs;
   
//     // std::cout << "Observation " << Observation << std::endl;
// }

// Eigen::Matrix<double, 1, 14> PolicyInterface::get_action()
// {
//     obs << Observation << std::endl;
//     for(int i = 0; i < 330; i++)
//         {Normalized_obs(i) = (Observation(i) - obs_mean(i)) / sqrt(obs_variance(i) + 1e-12);}
    
//     // std::cout << "normalized obs " << Normalized_obs << std::endl;
//     Eigen::Matrix<double, 1, 32> layer_1_out = (Normalized_obs * layer_1_weights + layer_1_bias).array().tanh();
//     Eigen::Matrix<double, 1, 512> layer_2_out = (layer_1_out * layer_2_weights + layer_2_bias).array().tanh();
//     Eigen::Matrix<double, 1, 512> layer_3_out = (layer_2_out * layer_3_weights + layer_3_bias).array().tanh();
//     Eigen::Matrix<double, 1, 32> layer_4_out = (layer_3_out * layer_4_weights + layer_4_bias).array().tanh();
//     Eigen::Matrix<double, 1, 28> layer_5_out = layer_4_out * layer_5_weights + layer_5_bias;

//     return action_scaler(layer_5_out.block(0, 0, 1, 14));
// }

// Eigen::Matrix<double, 1, 14> PolicyInterface::action_scaler(Eigen::Matrix<double, 1, 14> action)
// {
//     Eigen::Matrix<double, 1, 14> output;
//     for(int i = 0; i < 14; i++)
//     {
//         if(action(i) > 1.0)
//             action(i) = 1.0;
//         if(action(i) < -1.0)
//             action(i) = -1.0;

//     output(i) = action_lb(i) + 0.5 * (action(i) + 1.0) * (action_ub(i) - action_lb(i));
//     }
//     action_file << output << std::endl;
//     return output;
//     }

// this function is hardcoded for testing, must change it later to adapt different setups
void PolicyInterface::read_file(std::string filename) 
{
    // FILE* fp = fopen("output_no_MDC.json", "r");
    // FILE* fp = fopen("output_policy_random_state.json", "r");
    // FILE* fp = fopen("output_policy_test_1.json", "r");
    FILE* fp = fopen(filename.c_str(), "r");

    if(!fp)
    {
        std::cout << "Fail to open file" << std::endl;
        return;
    }

    char readBuffer[65536];
    rapidjson::FileReadStream is(fp, readBuffer,
                                 sizeof(readBuffer));
  
    // Parse the JSON document
    rapidjson::Document doc;
    doc.ParseStream(is);
  
    // Check if the document is valid
    if (doc.HasParseError()) {
        std::cerr << "Error: failed to parse JSON document"
                  << std::endl;
        fclose(fp);
        return;
    }
  
    // Close the file
    fclose(fp);
    // current implementation only works for 4 layer mlp
    const rapidjson::Value& obs_mean_array = doc["obs_mean"];
    const rapidjson::Value& obs_variance_array = doc["obs_variance"];
    obs_mean.resize(Eigen::NoChange, num_obs*5);
    obs_variance.resize(Eigen::NoChange, num_obs*5);

    for(int i = 0; i < num_obs*5; i++)
    {
        obs_mean(i) = obs_mean_array[i].GetDouble();
        obs_variance(i) = obs_variance_array[i].GetDouble();
    }


    const rapidjson::Value& layer1_weight_shape = doc["layer0/shape"];
    const rapidjson::Value& layer1_weight_value = doc["layer0/value"];
    const rapidjson::Value& layer1_bias_shape = doc["layer1/shape"];
    const rapidjson::Value& layer1_bias_value = doc["layer1/value"];
    Observation.resize(Eigen::NoChange, layer1_weight_shape[0].GetInt());
    Normalized_obs.resize(Eigen::NoChange, layer1_weight_shape[0].GetInt());
    layer_1_weights.resize(layer1_weight_shape[0].GetInt(), layer1_weight_shape[1].GetInt());
    layer_1_bias.resize(Eigen::NoChange, layer1_bias_shape[0].GetInt());
    for(int i = 0; i < layer1_weight_shape[0].GetInt(); i++)
    {
        for(int j = 0; j < layer1_weight_shape[1].GetInt(); j++)
        {
            layer_1_weights(i, j) = layer1_weight_value[i][j].GetDouble();
        }
    }
    for(int i = 0; i < layer1_bias_shape[0].GetInt(); i++)
    {
        layer_1_bias(i) = layer1_bias_value[i].GetDouble();
    }

    const rapidjson::Value& layer2_weight_shape = doc["layer2/shape"];
    const rapidjson::Value& layer2_weight_value = doc["layer2/value"];
    const rapidjson::Value& layer2_bias_shape = doc["layer3/shape"];
    const rapidjson::Value& layer2_bias_value = doc["layer3/value"];
    layer_2_weights.resize(layer2_weight_shape[0].GetInt(), layer2_weight_shape[1].GetInt());
    layer_2_bias.resize(Eigen::NoChange, layer2_bias_shape[0].GetInt());
    for(int i = 0; i < layer2_weight_shape[0].GetInt(); i++)
    {
        for(int j = 0; j < layer2_weight_shape[1].GetInt(); j++)
        {
            layer_2_weights(i, j) = layer2_weight_value[i][j].GetDouble();
        }
    }
    for(int i = 0; i < layer2_bias_shape[0].GetInt(); i++)
    {
        layer_2_bias(i) = layer2_bias_value[i].GetDouble();
    }

    const rapidjson::Value& layer3_weight_shape = doc["layer4/shape"];
    const rapidjson::Value& layer3_weight_value = doc["layer4/value"];
    const rapidjson::Value& layer3_bias_shape = doc["layer5/shape"];
    const rapidjson::Value& layer3_bias_value = doc["layer5/value"];
    layer_3_weights.resize(layer3_weight_shape[0].GetInt(), layer3_weight_shape[1].GetInt());
    layer_3_bias.resize(Eigen::NoChange, layer3_bias_shape[0].GetInt());
    for(int i = 0; i < layer3_weight_shape[0].GetInt(); i++)
    {
        for(int j = 0; j < layer3_weight_shape[1].GetInt(); j++)
        {
            layer_3_weights(i, j) = layer3_weight_value[i][j].GetDouble();
        }
    }
    for(int i = 0; i < layer3_bias_shape[0].GetInt(); i++)
    {
        layer_3_bias(i) = layer3_bias_value[i].GetDouble();
    }

    const rapidjson::Value& layer4_weight_shape = doc["layer6/shape"];
    const rapidjson::Value& layer4_weight_value = doc["layer6/value"];
    const rapidjson::Value& layer4_bias_shape = doc["layer7/shape"];
    const rapidjson::Value& layer4_bias_value = doc["layer7/value"];
    layer_4_weights.resize(layer4_weight_shape[0].GetInt(), layer4_weight_shape[1].GetInt());
    layer_4_bias.resize(Eigen::NoChange, layer4_bias_shape[0].GetInt());
    for(int i = 0; i < layer4_weight_shape[0].GetInt(); i++)
    {
        for(int j = 0; j < layer4_weight_shape[1].GetInt(); j++)
        {
            layer_4_weights(i, j) = layer4_weight_value[i][j].GetDouble();
        }
    }
    for(int i = 0; i < layer4_bias_shape[0].GetInt(); i++)
    {
        layer_4_bias(i) = layer4_bias_value[i].GetDouble();
    }
    //std::cout << layer_1_weights << std::endl;
    // const rapidjson::Value& layer5_weight_shape = doc["layer8/shape"];
    // const rapidjson::Value& layer5_weight_value = doc["layer8/value"];
    // const rapidjson::Value& layer5_bias_shape = doc["layer9/shape"];
    // const rapidjson::Value& layer5_bias_value = doc["layer9/value"];
    // layer_5_weights.resize(layer5_weight_shape[0].GetInt(), layer5_weight_shape[1].GetInt());
    // layer_5_bias.resize(Eigen::NoChange, layer5_bias_shape[0].GetInt());
    // for(int i = 0; i < layer5_weight_shape[0].GetInt(); i++)
    // {
    //     for(int j = 0; j < layer5_weight_shape[1].GetInt(); j++)
    //     {
    //         layer_5_weights(i, j) = layer5_weight_value[i][j].GetDouble();
    //     }
    // }
    // for(int i = 0; i < layer5_bias_shape[0].GetInt(); i++)
    // {
    //     layer_5_bias(i) = layer5_bias_value[i].GetDouble();
    // }

}
