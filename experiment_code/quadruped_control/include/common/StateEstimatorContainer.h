/*!
 * @file StateEstimator.h
 * @brief Implementation of State Estimator Interface
 *
 * Each StateEstimator object contains a number of estimators
 *
 * When the state estimator is run, it runs all estimators.
 */


#ifndef PROJECT_STATE_ESTIMATOR_CONTAINER_H
#define PROJECT_STATE_ESTIMATOR_CONTAINER_H

#pragma once

#include "LegController.h"
#include "../messages/LowlevelState.h"

/*!
 * "Cheater" state sent to robot from simulation
 */
struct CheaterState{
  Quat<double> orientation;
  Vec3<double> position;
  Vec3<double> omegaBody;
  Vec3<double> vBody;
  Vec3<double> acceleration;
};

/*!
 * Result of state estimation
 */

struct StateEstimate {
    Vec4<double> contactEstimate;
    Vec3<double> position;
    Vec3<double> vBody;
    Quat<double> orientation;
    Vec3<double> omegaBody;
    RotMat<double> rBody;
    Vec3<double> rpy;

    Vec3<double> omegaWorld;
    Vec3<double> vWorld;
    Vec3<double> aBody, aWorld;
};


/*!
 * input for state estimation
 */ 
struct StateEstimatorData {
    StateEstimate* result;
    LowlevelState* lowState;
    LegControllerData* legControllerData;
    Vec4<double>* contactPhase;
};

/*!
 * All Estimators inherit from this class
 */ 
class GenericEstimator{
  public:
    virtual void run() = 0;
    virtual void setup() = 0;

    void setData(StateEstimatorData data) {_stateEstimatorData = data;};

    virtual ~GenericEstimator() = default;
    StateEstimatorData _stateEstimatorData;
};

/*!
 * Main State Estimator class
 * Contains all GenericEstimator
 */ 
class StateEstimatorContainer {
  public:
    // Constructor
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateEstimatorContainer(LowlevelState *_lowState,
                            LegControllerData *_legControllerData,
                            StateEstimate *stateEstimate){
        _data.lowState = _lowState;
        _data.legControllerData = _legControllerData;
        _data.result = stateEstimate;
        _phase = Vec4<double>::Zero();
        _data.contactPhase = &_phase;
    }
    // deconstructor
    ~StateEstimatorContainer() {
        for (auto estimator : _estimators) {
            delete estimator;
        }
    }
    // run estimator
    void run(){
        for (auto estimator : _estimators){
            estimator->run();
        }
    }

    // get result
    const StateEstimate&  getResult() {return *_data.result;}

    // set contact phase
    void setContactPhase(Vec4<double>& phase) { *_data.contactPhase = phase; }

    // add estimator of given type
    template <typename EstimatorToAdd>
    void addEstimator(){
        std::cout << "add estimator" << std::endl;
        auto* estimator = new EstimatorToAdd();
        estimator->setData(_data);
        estimator->setup();
        _estimators.push_back(estimator);
    }

    // remove estimator of given type
    template <typename EstimatorToRemove>
    void removeEstimator() {
        int nRemoved = 0;
        _estimators.erase(
            std::remove_if(_estimators.begin(), _estimators.end(),
                           [&nRemoved](GenericEstimator* e){
                               if (dynamic_cast<EstimatorToRemove*>(e)){
                                   delete e;
                                   nRemoved++;
                                   return true;
                                } else {
                                    return false;
                                }
                           }),
            _estimators.end());
    }

    // remove all estimators
    void removeAllEstimators() {
        for (auto estimator : _estimators) {
            delete estimator;
        }
        _estimators.clear();
    }
  private:
    std::vector<GenericEstimator*> _estimators;
    Vec4<double> _phase;
    StateEstimatorData _data;
};


#endif