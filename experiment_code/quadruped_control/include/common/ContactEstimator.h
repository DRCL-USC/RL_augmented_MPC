/*! @file ContactEstimator.h
 *  @brief All Contact Estimation Algorithms
 *
 *  This file will contain all contact detection algorithms. For now, it just
 * has a pass-through algorithm which passes the phase estimation to the state
 * estimator.  This will need to change once we move contact detection to C++
 *
 *  We also still need to establish conventions for "phase" and "contact".
 */

#ifndef PROJECT_CONTACTESTIMATOR_H
#define PROJECT_CONTACTESTIMATOR_H

#include "StateEstimatorContainer.h"


/*!
 * A "passthrough" contact estimator which returns the expected contact state
 */
class ContactEstimator : public GenericEstimator {
 public:

  /*!
   * Set the estimated contact by copying the exptected contact state into the
   * estimated contact state
   */
  virtual void run() {
    // this->_stateEstimatorData.result->contactEstimate =
    //     *this->_stateEstimatorData.contactPhase;

    for(int i = 0; i < 4; i++){
      if(this->_stateEstimatorData.lowState->FootForce[i] > 10){
        this->_stateEstimatorData.result->contactEstimate(i) = 0.5;
      }

      else{
        this->_stateEstimatorData.result->contactEstimate(i) = 0;
      }
    }
  }

  /*!
   * Set up the contact estimator
   */
  virtual void setup() {}
};

#endif  // PROJECT_CONTACTESTIMATOR_H