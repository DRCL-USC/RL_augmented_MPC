#ifndef POLICY_INTERFACE
#define POLICY_INTERFACE

// a base class to import netowrk weights and bias

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include <fstream>
#include <sstream>
#include <tuple>     
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using Eigen::Dynamic;

class PolicyInterface
{
  public:
    PolicyInterface(int _num_obs, int _num_action, std::string _filename);
    ~PolicyInterface(){};

    void read_file(std::string filename);
    // Eigen::Matrix<double, 1, 14> get_action();
    // void updateObservation(Eigen::Matrix<double, 1, 66> current_obs);
    // Eigen::Matrix<double, 1, 14> action_scaler(Eigen::Matrix<double, 1, 14> action);

    // std::ofstream obs;
    // std::ofstream action_file;

    int num_obs;
    int num_action;

  protected:
  
    Eigen::Matrix<double, 1, Dynamic> Observation;
    Eigen::Matrix<double, 1, Dynamic> Normalized_obs;
    Eigen::Matrix<double, Dynamic, Dynamic> layer_1_weights;
    Eigen::Matrix<double, 1, Dynamic> layer_1_bias;

    Eigen::Matrix<double, Dynamic, Dynamic> layer_2_weights;
    Eigen::Matrix<double, 1, Dynamic> layer_2_bias;

    Eigen::Matrix<double, Dynamic, Dynamic> layer_3_weights;
    Eigen::Matrix<double, 1, Dynamic> layer_3_bias;

    Eigen::Matrix<double, Dynamic, Dynamic> layer_4_weights;
    Eigen:: Matrix<double,1, Dynamic> layer_4_bias;

    Eigen::Matrix<double, 1, Dynamic> obs_mean;
    Eigen::Matrix<double, 1, Dynamic> obs_variance;

};

#endif