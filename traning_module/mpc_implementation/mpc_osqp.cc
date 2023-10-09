// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <assert.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <fstream>
#define DCHECK_GT(a, b) assert((a) > (b))
#define DCHECK_EQ(a, b) assert((a) == (b))

#ifdef _WIN32
typedef __int64 qp_int64;
#else
typedef long long qp_int64;
#endif  //_WIN32

using Eigen::AngleAxisd;
using Eigen::Map;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;
#include "qpOASES.hpp"
#include "qpOASES/Types.hpp"

using qpOASES::QProblem;

typedef Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>
    RowMajorMatrixXd;

constexpr int k3Dim = 3;
constexpr double kGravity = 9.8;
constexpr double kMaxScale = 3.0;
constexpr double kMinScale = 0.0;

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "osqp/include/ctrlc.h"
#include "osqp/include/osqp.h"

enum QPSolverName { OSQP, QPOASES };

bool near_zero(qpOASES::real_t a) { return (a < 0.01 && a > -.01); }

bool near_one(qpOASES::real_t a) { return near_zero(a - 1); }

// Auxiliary function for copying data to qpOASES data structure.
void CopyToVec(const Eigen::VectorXd& vec,
               const Eigen::MatrixXd& foot_contact_states, int num_legs,
               int planning_horizon, int blk_size,
               std::vector<qpOASES::real_t>* out) {
  int buffer_index = 0;
  for (int i = 0; i < num_legs * planning_horizon; i++) {
    // otherwise copy this block.
    assert(buffer_index < out->size());
    for (int j = 0; j < blk_size; ++j) {
      int index = i * blk_size + j;
      (*out)[buffer_index] = vec[index];
      buffer_index++;
    }
  }
}

// Auxiliary function for copying data to qpOASES data structure.
void CopyToMatrix(const Eigen::MatrixXd& input,
                  const Eigen::MatrixXd& foot_contact_states, int num_legs,
                  int planning_horizon, int row_blk_size, int col_blk_size,
                  bool is_block_diagonal, Eigen::Map<RowMajorMatrixXd>* out) {
  // the block index in the destination matrix.
  int row_blk = 0;
  for (int i = 0; i < planning_horizon * num_legs; ++i) {
    if (is_block_diagonal) {
      // just copy the block
      int col_blk = row_blk;
      out->block(row_blk * row_blk_size, col_blk * col_blk_size, row_blk_size,
                 col_blk_size) = input.block(i * row_blk_size, i * col_blk_size,
                                             row_blk_size, col_blk_size);
    } else {
      int col_blk = 0;
      // Non-diagonal, need to copy all elements.
      for (int j = 0; j < planning_horizon * num_legs; ++j) {
        out->block(row_blk * row_blk_size, col_blk * col_blk_size, row_blk_size,
                   col_blk_size) =
            input.block(i * row_blk_size, j * col_blk_size, row_blk_size,
                        col_blk_size);
        ++col_blk;
      }
    }
    ++row_blk;
  }
}

// Converts the roll pitchh yaw angle vector to the corresponding rotation
// matrix.
Eigen::Matrix3d ConvertRpyToRot(const Eigen::Vector3d& rpy);

// Converts a vector to the skew symmetric matrix form. For an input vector
// [a, b, c], the output matrix would be:
//   [ 0, -c,  b]
//   [ c,  0, -a]
//   [-b,  a,  0]
Eigen::Matrix3d ConvertToSkewSymmetric(const Eigen::Vector3d& vec);

// The CoM dynamics can be written as:
//   X_dot = A X + B u
// where X is the 13-dimensional state vector (r, p, y, x, y, z, r_dot, p_dot,
// y_dot, vx, vy, vz, -g) constructed from the CoM roll/pitch/yaw/position, and
// their first order derivatives. 'g' is the gravity constant. This API
// constructs the A matrix in the formula. Check the MIT paper for details of
// the formulation.
void CalculateAMat(const Eigen::Vector3d& rpy,
                   const Eigen::Vector3d& desired_linear_accel,
                   const Eigen::Vector3d& desired_angular_accel,
                   Eigen::MatrixXd* a_mat_ptr);

// Constructs the B matrix in the linearized CoM dynamics equation. See the
// documentation for 'CalculateAMat' for details of the symbol.
void CalculateBMat(double inv_mass, const Eigen::Matrix3d& inv_inertia,
                   const Eigen::MatrixXd& foot_positions,
                   Eigen::MatrixXd* b_mat_ptr);

// Calculates the discretized space-time dynamics. Given the dynamics equation:
//   X_dot = A X + B u
// and a timestep dt, we can estimate the snapshot of the state at t + dt by:
//   X[t + dt] = exp([A, B]dt) [X, u] = A_exp X + B_exp u
void CalculateExponentials(const Eigen::MatrixXd& a_mat,
                           const Eigen::MatrixXd& b_mat, double timestep,
                           Eigen::MatrixXd* ab_mat_ptr,
                           Eigen::MatrixXd* a_exp_ptr,
                           Eigen::MatrixXd* b_exp_ptr);

// Calculates the dense QP formulation of the discretized space time dynamics.
// Given:
//   X_k+1 = A_exp X_k + B_exp u_k
// We can unroll the dynamics in time using a forward pass:
//   [X_1, X_2,..., X_k+1] = A_qp X_0 + B_qp [u_0, u_1,..., u_k]
void CalculateQpMats(const Eigen::MatrixXd& a_exp, const Eigen::MatrixXd& b_exp,
                     int horizon, Eigen::MatrixXd* a_qp_ptr, MatrixXd* anb_aux_ptr,
                     Eigen::MatrixXd* b_qp_ptr, MatrixXd* p_mat_ptr);

void UpdateConstraintsMatrix(std::vector<double>& friction_coeff, int horizon,
                             int num_legs, Eigen::MatrixXd* constraint_ptr);

void CalculateConstraintBounds(const Eigen::MatrixXd& contact_state,
                               double fz_max, double fz_min,
                               double friction_coeff, int horizon,
                               Eigen::VectorXd* constraint_lb_ptr,
                               Eigen::VectorXd* constraint_ub_ptr);

double EstimateCoMHeightSimple(const Eigen::MatrixXd& foot_positions_world,
                               const Eigen::MatrixXd& foot_contact_states);

// The MIT convex mpc implementation as described in this paper:
//   https://ieeexplore.ieee.org/document/8594448/
// Computes the optimal feet contact forces given a desired center of mass
// trajectory and gait pattern.
class ConvexMpc {
 public:
  static constexpr int kStateDim =
      13;  // 6 dof pose + 6 dof velocity + 1 gravity.

  // For each foot contact force we use 4-dim cone approximation + 1 for z.
  static constexpr int kConstraintDim = 5;

  ConvexMpc(double mass, const std::vector<double>& inertia, int num_legs,
            int planning_horizon, double timestep,
            // const std::vector<double>& qp_weights, 
            double alpha = 1e-5,
            QPSolverName qp_solver_name = QPOASES);

  virtual ~ConvexMpc() { osqp_cleanup(workspace_); }
  // If not explicitly specified, we assume the quantities are measured in a
  // world frame. Usually we choose the yaw-aligned horizontal frame i.e. an
  // instanteneous world frame at the time of planning with its origin at CoM
  // and z axis aligned with gravity. The yaw-alignment means that the CoM
  // rotation measured in this frame has zero yaw component. Caveat: We expect
  // the input euler angle roll_pitch_yaw to be in ZYX format, i.e. the rotation
  // order is X -> Y -> Z, with respect to the extrinsic (fixed) coordinate
  // frame. In the intrinsic (body-attached) frame the rotation order is Z -> Y'
  // -> X".
  std::vector<double> ComputeContactForces(
      const std::vector<double>& qp_weights,
      std::vector<double> com_position,
      std::vector<double> com_velocity,
      std::vector<double> com_roll_pitch_yaw,
      std::vector<double> ground_normal_vec,
      std::vector<double> com_angular_velocity,
      std::vector<double> foot_contact_states,
      std::vector<double> foot_positions_world_frame,
      std::vector<double> foot_friction_coeffs,
      std::vector<double> desired_com_position,
      std::vector<double> desired_com_velocity,
      std::vector<double> desired_com_roll_pitch_yaw,
      std::vector<double> desired_com_angular_velocity);

  // Reset the solver so that for the next optimization run the solver is
  // re-initialized.
  void ResetSolver();
  std::ofstream mpc_data;
 private:
  const double mass_;
  const double inv_mass_;
  const Eigen::Matrix3d inertia_;
  const Eigen::Matrix3d inv_inertia_;
  const int num_legs_;
  const int planning_horizon_;
  const double timestep_;
  QPSolverName qp_solver_name_;

  // 13 * horizon diagonal matrix.
  Eigen::MatrixXd qp_weights_;

  // 13 x 13 diagonal matrix.
  Eigen::MatrixXd qp_weights_single_;

  // num_legs * 3 * horizon diagonal matrix.
  const Eigen::MatrixXd alpha_;
  const Eigen::MatrixXd alpha_single_;
  const int action_dim_;

  // The following matrices will be updated for every call. However, their sizes
  // can be determined at class initialization time.
  Eigen::VectorXd state_;                 // 13
  Eigen::VectorXd desired_states_;        // 13 * horizon
  Eigen::MatrixXd contact_states_;        // horizon x num_legs
  Eigen::MatrixXd foot_positions_base_;   // num_legs x 3
  Eigen::MatrixXd foot_positions_world_;  // num_legs x 3
  Eigen::VectorXd foot_friction_coeff_;   // num_legs
  Eigen::Matrix3d rotation_;
  Eigen::Matrix3d inertia_world_;    // rotation x inertia x rotation_transpose
  Eigen::MatrixXd a_mat_;            // 13 x 13
  Eigen::MatrixXd b_mat_;            // 13 x 12
  Eigen::MatrixXd ab_concatenated_;  // 13 + num_legs * 3 x 13 + num_legs * 3
  Eigen::MatrixXd a_exp_;            // same dimension as a_mat_
  Eigen::MatrixXd b_exp_;            // same dimension as b_mat_


  // Contains all the power mats of a_exp_. Consider Eigen::SparseMatrix.
  Eigen::MatrixXd a_qp_;  // 13 * horizon x 13
  Eigen::MatrixXd b_qp_;  // 13 * horizon x num_legs * 3 * horizon sparse
  Eigen::MatrixXd b_qp_transpose_;
  Eigen::MatrixXd p_mat_;  // num_legs * 3 * horizon x num_legs * 3 * horizon
  Eigen::VectorXd q_vec_;  // num_legs * 3 * horizon vector

  // Auxiliary containing A^n*B, with n in [0, num_legs * 3)
  Eigen::MatrixXd anb_aux_;  // 13 * horizon x (num_legs * 3)

  // Contains the constraint matrix and bounds.
  Eigen::MatrixXd
      constraint_;  // 5 * num_legs * horizon x 3 * num_legs * horizon
  Eigen::VectorXd constraint_lb_;  // 5 * num_legs * horizon
  Eigen::VectorXd constraint_ub_;  // 5 * num_legs * horizon

  std::vector<double> qp_solution_;

  ::OSQPWorkspace* workspace_;
  // Whether optimizing for the first step
  bool initial_run_;
};

constexpr int ConvexMpc::kStateDim;

Matrix3d ConvertRpyToRot(const Vector3d& rpy) {
  assert(rpy.size() == k3Dim);
  const AngleAxisd roll(rpy[0], Vector3d::UnitX());
  const AngleAxisd pitch(rpy[1], Vector3d::UnitY());
  const AngleAxisd yaw(rpy[2], Vector3d::UnitZ());
  Quaterniond q = yaw * pitch * roll;

  return q.matrix();
}

Matrix3d ConvertToSkewSymmetric(const Vector3d& vec) {
  Matrix3d skew_symm;
  skew_symm << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
  return skew_symm;
}

void CalculateAMat(const Vector3d& rpy, const Eigen::Vector3d& desired_linear_accel,
                   const Eigen::Vector3d& desired_angular_accel,
                   MatrixXd* a_mat_ptr) {
  // The transformation of angular velocity to roll pitch yaw rate. Caveat:
  // rpy rate is not a proper vector and does not follow the common vector
  // transformation dicted by the rotation matrix. Here we assume the input
  // rotation is in X->Y->Z order in the extrinsic/fixed frame, or z->y'->x''
  // order in the intrinsic frame.
  const double cos_yaw = cos(rpy[2]);
  const double sin_yaw = sin(rpy[2]);
  const double cos_pitch = cos(rpy[1]);
  const double tan_pitch = tan(rpy[1]);
  Matrix3d angular_velocity_to_rpy_rate;
  // angular_velocity_to_rpy_rate << cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0,
  //       -sin_yaw, cos_yaw, 0, cos_yaw* tan_pitch, sin_yaw* tan_pitch, 1;
  angular_velocity_to_rpy_rate << cos_yaw , -sin_yaw, 0,
                                  sin_yaw, cos_yaw, 0, 
                                  0, 0, 1;

  MatrixXd& a_mat = *a_mat_ptr;
  a_mat.block<3, 3>(0, 6) = angular_velocity_to_rpy_rate.transpose();
  a_mat(3, 9) = 1;
  a_mat(4, 10) = 1;
  a_mat(5, 11) = 1;
  a_mat(6, 12) = desired_angular_accel[0];
  a_mat(7, 12) = desired_angular_accel[1];
  a_mat(8, 12) = desired_angular_accel[2];
  a_mat(9, 12) = desired_linear_accel[0];
  a_mat(10, 12) = desired_linear_accel[1];
  a_mat(11, 12) = desired_linear_accel[2];
  a_mat(12, 12) = 0;

}

void CalculateBMat(double inv_mass, const Matrix3d& inv_inertia,
                   const MatrixXd& foot_positions, MatrixXd* b_mat_ptr) {
  // b_mat contains non_zero elements only in row 6:12.
  const int num_legs = foot_positions.rows();
  MatrixXd& b_mat = *b_mat_ptr;
  for (int i = 0; i < num_legs; ++i) {
    b_mat.block<k3Dim, k3Dim>(6, i * k3Dim) =
        inv_inertia * ConvertToSkewSymmetric(foot_positions.row(i));
    b_mat(9, i * k3Dim) = inv_mass;
    b_mat(10, i * k3Dim + 1) = inv_mass;
    b_mat(11, i * k3Dim + 2) = inv_mass;
  }
}

void CalculateExponentials(const MatrixXd& a_mat, const MatrixXd& b_mat,
                           double timestep, MatrixXd* ab_mat_ptr,
                           MatrixXd* a_exp_ptr, MatrixXd* b_exp_ptr) {
  MatrixXd& ab_mat = *ab_mat_ptr;
  ab_mat.block(0, 0, 13, 13) = a_mat;
  ab_mat.block(0, 13, 13, 12) = b_mat;
  ab_mat *= timestep;
  // This temporary is inevitable.
  MatrixXd ab_exp = ab_mat.exp();
  *a_exp_ptr = ab_exp.block(0, 0, 13, 13);
  *b_exp_ptr = ab_exp.block(0, 13, 13, 12);
}

void CalculateQpMats(const MatrixXd& a_exp, const MatrixXd& b_exp,
    const MatrixXd& qp_weights_single,
    const MatrixXd& alpha_single, int horizon,
    MatrixXd* a_qp_ptr, MatrixXd* anb_aux_ptr,
    MatrixXd* b_qp_ptr, MatrixXd* p_mat_ptr) {
    const int state_dim = ConvexMpc::kStateDim;
    MatrixXd& a_qp = *a_qp_ptr;
    a_qp.block(0, 0, state_dim, state_dim) = a_exp;
    for (int i = 1; i < horizon; i++) {
        a_qp.block<state_dim, state_dim>(i * state_dim, 0) =
            a_exp * a_qp.block<state_dim, state_dim>((i - 1) * state_dim, 0);
    }

    const int action_dim = b_exp.cols();

    MatrixXd& anb_aux = *anb_aux_ptr;
    // Compute auxiliary matrix: [B_exp, A_exp * B_exp, ..., A_exp^(h-1) * B_exp]
    anb_aux.block(0, 0, state_dim, action_dim) = b_exp;
    for (int i = 1; i < horizon; i++) {
        anb_aux.block(i * state_dim, 0, state_dim, action_dim) =
            a_exp * anb_aux.block((i - 1) * state_dim, 0, state_dim, action_dim);
    }

    MatrixXd& b_qp = *b_qp_ptr;
    for (int i = 0; i < horizon; i++) {
        // Diagonal block.
        b_qp.block(i * state_dim, i * action_dim, state_dim, action_dim) = b_exp;
        // Off diagonal Diagonal block = A^(i - j - 1) * B_exp.
        for (int j = 0; j < i; j++) {
            const int power = i - j;
            b_qp.block(i * state_dim, j * action_dim, state_dim, action_dim) =
                anb_aux.block(power * state_dim, 0, state_dim, action_dim);
        }
    }

    // We construct the P matrix by filling in h x h submatrices, each with size
    // action_dim x action_dim.
    // The r_th (r in [1, h]) diagonal submatrix of P is:
    // 2 * sum_{i=0:h-r}(B'A'^i L A^i B) + alpha, where h is the horizon.
    // The off-diagonal submatrix at row r and column c of P is:
    // 2 * sum_{i=0:h-c}(B'A'^{h-r-i} L A^{h-c-i} B)
    MatrixXd& p_mat = *p_mat_ptr;
    // We first compute the submatrices at column h.
    for (int i = horizon - 1; i >= 0; i--) {
        p_mat.block(i * action_dim, (horizon - 1) * action_dim, action_dim,
            action_dim) =
            anb_aux.block((horizon - i - 1) * state_dim, 0, state_dim, action_dim)
            .transpose() *
            qp_weights_single * b_exp;
       // Fill the lower-triangle part by transposing the corresponding
       // upper-triangle part.
        if (i != horizon - 1) {
            p_mat.block((horizon - 1) * action_dim, i * action_dim, action_dim,
                action_dim) =
                p_mat
                .block(i * action_dim, (horizon - 1) * action_dim, action_dim,
                    action_dim)
                .transpose();
        }
    }

    // We then fill in the submatrices in the middle by propagating the values
    // from lower right to upper left.
    for (int i = horizon - 2; i >= 0; i--) {
        // Diagonal block.
        p_mat.block(i * action_dim, i * action_dim, action_dim, action_dim) =
            p_mat.block((i + 1) * action_dim, (i + 1) * action_dim, action_dim,
                action_dim) +
            anb_aux.block((horizon - i - 1) * state_dim, 0, state_dim, action_dim)
            .transpose() *
            qp_weights_single *
            anb_aux.block((horizon - i - 1) * state_dim, 0, state_dim,
                action_dim);
        // Off diagonal block
        for (int j = i + 1; j < horizon - 1; j++) {
            p_mat.block(i * action_dim, j * action_dim, action_dim, action_dim) =
                p_mat.block((i + 1) * action_dim, (j + 1) * action_dim, action_dim,
                    action_dim) +
                anb_aux.block((horizon - i - 1) * state_dim, 0, state_dim, action_dim)
                .transpose() *
                qp_weights_single *
                anb_aux.block((horizon - j - 1) * state_dim, 0, state_dim,
                    action_dim);
            // Fill the lower-triangle part by transposing the corresponding
            // upper-triangle part.
            p_mat.block(j * action_dim, i * action_dim, action_dim, action_dim) =
                p_mat.block(i * action_dim, j * action_dim, action_dim, action_dim)
                .transpose();
        }
    }

    // Multiply by 2 and add alpha.
    p_mat *= 2.0;
    for (int i = 0; i < horizon; i++) {
        p_mat.block(i * action_dim, i * action_dim, action_dim, action_dim) +=
            alpha_single;
    }
}

void UpdateConstraintsMatrix(double friction_coeff, int horizon,
                             int num_legs, MatrixXd* constraint_ptr) {
  const int constraint_dim = ConvexMpc::kConstraintDim;
  MatrixXd& constraint = *constraint_ptr;
  double mu = 1.0/friction_coeff;
  for (int i = 0; i < horizon * num_legs; ++i) {
    constraint.block<constraint_dim, k3Dim>(i * constraint_dim, i * k3Dim)  << 
    mu, 0,  1.f,
    -mu, 0,  1.f,
    0,  mu, 1.f,
    0, -mu, 1.f,
    0,   0, 1.f;
    // -1, 0, friction_coeff[0], 
    //       1, 0, friction_coeff[1], 
    //       0, -1, friction_coeff[2],
    //     0, 1, friction_coeff[3], 
    //     0, 0, 1;
    } 
}

void CalculateConstraintBounds(const MatrixXd& contact_state, double fz_max,
                               double fz_min, double friction_coeff,
                               int horizon, VectorXd* constraint_lb_ptr,
                               VectorXd* constraint_ub_ptr) {
  const int constraint_dim = ConvexMpc::kConstraintDim;

  const int num_legs = contact_state.cols();

  VectorXd& constraint_lb = *constraint_lb_ptr;
  VectorXd& constraint_ub = *constraint_ub_ptr;
  for (int i = 0; i < horizon; ++i) {
    for (int j = 0; j < num_legs; ++j) {
      const int row = (i * num_legs + j) * constraint_dim;
      constraint_lb(row) = 0;
      constraint_lb(row + 1) = 0;
      constraint_lb(row + 2) = 0;
      constraint_lb(row + 3) = 0;
      constraint_lb(row + 4) = fz_min * contact_state(i, j);

      const double friction_ub =
          (friction_coeff) * fz_max * contact_state(i, j);
      constraint_ub(row) = 5e10; //friction_ub;
      constraint_ub(row + 1) = 5e10; //friction_ub;
      constraint_ub(row + 2) = 5e10; //friction_ub;
      constraint_ub(row + 3) = 5e10; //friction_ub;
      constraint_ub(row + 4) = fz_max * contact_state(i, j);
    }
  }
}

double EstimateCoMHeightSimple(const MatrixXd& foot_positions_world,
                               const Eigen::MatrixXd& foot_contact_states) {
  int legs_in_contact = 0;
  double com_height = 0;
  const int z_dim = 2;
  for (int i = 0; i < foot_contact_states.size(); ++i) {
    if (foot_contact_states(i)) {
      com_height += foot_positions_world(i, z_dim);
      legs_in_contact += 1;
    }
  }

  // We don't support jumping in air for now.
  DCHECK_GT(legs_in_contact, 0);
  return abs(com_height / legs_in_contact);
}

MatrixXd AsBlockDiagonalMat(const std::vector<double>& qp_weights,
                            int planning_horizon) {
  const Eigen::Map<const VectorXd> qp_weights_vec(qp_weights.data(),
                                                  qp_weights.size());
  // Directly return the rhs will cause a TSAN failure, probably due to the
  // asDiagonal not reall copying the memory. Creates the temporary will ensure
  // copy on return.
  const MatrixXd qp_weights_mat =
      qp_weights_vec.replicate(planning_horizon, 1).asDiagonal();
  return qp_weights_mat;
}

ConvexMpc::ConvexMpc(double mass, const std::vector<double>& inertia,
                     int num_legs, int planning_horizon, double timestep,
                     double alpha,
                     QPSolverName qp_solver_name)
    : mass_(mass),
      inv_mass_(1 / mass),
      inertia_(inertia.data()),
      inv_inertia_(inertia_.inverse()),
      num_legs_(num_legs),
      planning_horizon_(planning_horizon),
      timestep_(timestep),
      qp_solver_name_(qp_solver_name),
      // qp_weights_(AsBlockDiagonalMat(qp_weights, planning_horizon)),
      // qp_weights_single_(AsBlockDiagonalMat(qp_weights, 1)),
      alpha_(alpha * MatrixXd::Identity(num_legs * planning_horizon * k3Dim,
                                        num_legs * planning_horizon * k3Dim)),
      alpha_single_(alpha *
                    MatrixXd::Identity(num_legs * k3Dim, num_legs * k3Dim)),
      action_dim_(num_legs * k3Dim),
      state_(kStateDim),
      desired_states_(kStateDim * planning_horizon),
      contact_states_(planning_horizon, num_legs),
      foot_positions_base_(num_legs, k3Dim),
      foot_positions_world_(num_legs, k3Dim),
      foot_friction_coeff_(num_legs_),
      a_mat_(kStateDim, kStateDim),
      b_mat_(kStateDim, action_dim_),
      ab_concatenated_(kStateDim + action_dim_, kStateDim + action_dim_),
      a_exp_(kStateDim, kStateDim),
      b_exp_(kStateDim, action_dim_),
      a_qp_(kStateDim * planning_horizon, kStateDim),
      b_qp_(kStateDim * planning_horizon, action_dim_ * planning_horizon),
      p_mat_(num_legs * planning_horizon * k3Dim,
             num_legs * planning_horizon * k3Dim),
      q_vec_(num_legs * planning_horizon * k3Dim),
      anb_aux_(kStateDim * planning_horizon, action_dim_),
      constraint_(kConstraintDim * num_legs * planning_horizon,
                  action_dim_ * planning_horizon),
      constraint_lb_(kConstraintDim * num_legs * planning_horizon),
      constraint_ub_(kConstraintDim * num_legs * planning_horizon),
      qp_solution_(k3Dim * num_legs * planning_horizon),
      workspace_(0),
      initial_run_(true)

{
  // assert(qp_weights.size() == kStateDim);
  // We assume the input inertia is a 3x3 matrix.
  assert(inertia.size() == k3Dim * k3Dim);
  state_.setZero();
  desired_states_.setZero();
  contact_states_.setZero();
  foot_positions_base_.setZero();
  foot_positions_world_.setZero();
  foot_friction_coeff_.setZero();
  a_mat_.setZero();
  b_mat_.setZero();
  ab_concatenated_.setZero();
  a_exp_.setZero();
  b_exp_.setZero();
  a_qp_.setZero();
  b_qp_.setZero();
  b_qp_transpose_.setZero();
  constraint_.setZero();
  constraint_lb_.setZero();
  constraint_ub_.setZero();
  // mpc_data.open("mpc.txt");
}

void ConvexMpc::ResetSolver() { initial_run_ = true; }

std::vector<double> ConvexMpc::ComputeContactForces(
    const std::vector<double>& qp_weights,
    std::vector<double> com_position,
    std::vector<double> com_velocity,
    std::vector<double> com_roll_pitch_yaw,
    std::vector<double> ground_normal_vec,
    std::vector<double> com_angular_velocity,
    std::vector<double> foot_contact_states_flattened,
    std::vector<double> foot_positions_world_frame,
    std::vector<double> foot_friction_coeffs,
    std::vector<double> desired_com_position,
    std::vector<double> desired_com_velocity,
    std::vector<double> desired_com_roll_pitch_yaw,
    std::vector<double> desired_com_angular_velocity){

  assert(qp_weights.size() == kStateDim);
  qp_weights_ = AsBlockDiagonalMat(qp_weights, planning_horizon_);
  qp_weights_single_ = AsBlockDiagonalMat(qp_weights, 1);

  std::vector<double> error_result;
  Eigen::MatrixXd foot_contact_states =
      Eigen::Map<const Eigen::MatrixXd>(foot_contact_states_flattened.data(),
                                        num_legs_, planning_horizon_).transpose();
  auto start = std::chrono::system_clock::now();

    // const Quaterniond com_rotation =
    //   AngleAxisd(com_roll_pitch_yaw[0], Vector3d::UnitX()) *
    //   AngleAxisd(com_roll_pitch_yaw[1], Vector3d::UnitY()) *
    //   AngleAxisd(com_roll_pitch_yaw[2], Vector3d::UnitZ());

  foot_positions_world_ = Eigen::Map<const MatrixXd>(
                             foot_positions_world_frame.data(), k3Dim, num_legs_).transpose();

  // foot_positions_base_ = Eigen::Map<const MatrixXd>(
  //                            foot_positions_body_frame.data(), k3Dim, num_legs_)
  //                            .transpose();
  // for (int i = 0; i < num_legs_; ++i) {
  //   foot_positions_world_.row(i) = com_rotation * foot_positions_base_.row(i);
  // }

  // std::cout << "foot pos \n" << foot_positions_world_ << std::endl;

  // Now we can estimate the body height using the world frame foot positions
  // and contact states. We use simple averges leg height here.
  // DCHECK_EQ(foot_contact_states[0].size(), num_legs_);
  const double com_x = com_position[0];
  const double com_y = com_position[1];
  const double com_z = com_position[2];

  // Prepare the current and desired state vectors of length kStateDim *
  // planning_horizon.
  DCHECK_EQ(com_velocity.size(), k3Dim);
  DCHECK_EQ(com_angular_velocity.size(), k3Dim);
  state_ << com_roll_pitch_yaw[0], com_roll_pitch_yaw[1], com_roll_pitch_yaw[2],
      com_x, com_y, com_z, com_angular_velocity[0], com_angular_velocity[1],
      com_angular_velocity[2], com_velocity[0], com_velocity[1],
      com_velocity[2], 1.0;

  for (int i = 0; i < planning_horizon_; i++) {
    desired_states_[i * kStateDim + 0] = desired_com_roll_pitch_yaw[0];
    desired_states_[i * kStateDim + 1] = desired_com_roll_pitch_yaw[1];
    desired_states_[i * kStateDim + 2] =
        desired_com_roll_pitch_yaw[2] +
        timestep_ * (i + 1) * desired_com_angular_velocity[2];

    desired_states_[i * kStateDim + 3] =
        timestep_ * (i + 1) * desired_com_velocity[0] + desired_com_position[0];
    desired_states_[i * kStateDim + 4] =
        timestep_ * (i + 1) * desired_com_velocity[1] + desired_com_position[1];
    desired_states_[i * kStateDim + 5] = desired_com_position[2];

    
    // Prefer to stablize roll and pitch.
    desired_states_[i * kStateDim + 6] = desired_com_angular_velocity[0];
    desired_states_[i * kStateDim + 7] = desired_com_angular_velocity[1];
    desired_states_[i * kStateDim + 8] = desired_com_angular_velocity[2];

    desired_states_[i * kStateDim + 9] = desired_com_velocity[0];
    desired_states_[i * kStateDim + 10] = desired_com_velocity[1];
    // Prefer to stablize the body height.
    desired_states_[i * kStateDim + 11] = 0;

    desired_states_[i * kStateDim + 12] = 1;
  }

  // std::cout << "desired states: " << desired_states_ << std::endl;

  const Vector3d rpy(com_roll_pitch_yaw[0], com_roll_pitch_yaw[1],
                     com_roll_pitch_yaw[2]);

  const Vector3d linear_accel(ground_normal_vec[3], ground_normal_vec[4],
                               ground_normal_vec[5]);
                            
  const Vector3d angular_accel(ground_normal_vec[0], ground_normal_vec[1],
                               ground_normal_vec[2]);

  CalculateAMat(rpy, linear_accel, angular_accel, &a_mat_);
  double cos_yaw = cos(rpy[2]);
  double sin_yaw = sin(rpy[2]);
  Matrix3d rotation_;
  // rotation_ = ConvertRpyToRot(rpy);
  rotation_ << cos_yaw , -sin_yaw,  0,
      sin_yaw, cos_yaw, 0, 0, 0 , 1;

  const Matrix3d inv_inertia_world =
      (rotation_ * inertia_* rotation_.transpose()).inverse();

  CalculateBMat(inv_mass_, inv_inertia_world, foot_positions_world_, &b_mat_);

  CalculateExponentials(a_mat_, b_mat_, timestep_, &ab_concatenated_, &a_exp_,
                        &b_exp_);
  // std::cout << "a_mat \n" << a_mat_ << std::endl;
  //  std::cout << "b_mat \n" << b_mat_ << std::endl;
  // std::cout << "a_exp \n" << a_exp_ << std::endl;

  CalculateQpMats(a_exp_, b_exp_, qp_weights_single_, alpha_single_,
                  planning_horizon_, &a_qp_, &anb_aux_, &b_qp_, &p_mat_);

  const MatrixXd state_diff = a_qp_ * state_ - desired_states_;

  // std::cout << "vx " << com_x << std::endl;
  // std::cout << "a qpoases  \n" <<  a_qp_ << std::endl;
  
  
  p_mat_ = 2 * (b_qp_.transpose() * qp_weights_ * b_qp_ + alpha_);
  q_vec_ = 2 * b_qp_.transpose() * (qp_weights_ * state_diff);

  CalculateConstraintBounds(foot_contact_states, mass_ * kGravity * kMaxScale,
                            mass_ * kGravity * kMinScale,
                            foot_friction_coeffs[0], planning_horizon_,
                            &constraint_lb_, &constraint_ub_);
  // std::cout << "constraint \n" << constraint_ub_ << std::endl;

    // Solve the QP Problem using qpOASES
    UpdateConstraintsMatrix(foot_friction_coeffs[0], planning_horizon_, num_legs_,
                            &constraint_);

    // To use qpOASES, we need to eleminate the zero rows/cols from the
    // matrices when copy to qpOASES buffer
    int num_legs_in_contact = 4;

    const int qp_dim = 12 * planning_horizon_;
    const int constraint_dim = 20 * planning_horizon_;
    std::vector<qpOASES::real_t> hessian(qp_dim * qp_dim, 0);
    Map<RowMajorMatrixXd> hessian_mat_view(hessian.data(), qp_dim, qp_dim);
    // Copy to the hessian
    CopyToMatrix(p_mat_, foot_contact_states, num_legs_, planning_horizon_,
                 k3Dim, k3Dim, false, &hessian_mat_view);

    std::vector<qpOASES::real_t> g_vec(qp_dim, 0);
    // Copy the g_vec
    CopyToVec(q_vec_, foot_contact_states, num_legs_, planning_horizon_, k3Dim,
              &g_vec);

    std::vector<qpOASES::real_t> a_mat(qp_dim * constraint_dim, 0);
    Map<RowMajorMatrixXd> a_mat_view(a_mat.data(), constraint_dim, qp_dim);
    CopyToMatrix(constraint_, foot_contact_states, num_legs_, planning_horizon_,
                 5, k3Dim, true, &a_mat_view);

    std::vector<qpOASES::real_t> a_lb(constraint_dim, 0);
    CopyToVec(constraint_lb_, foot_contact_states, num_legs_, planning_horizon_,
              5, &a_lb);

    std::vector<qpOASES::real_t> a_ub(constraint_dim, 0);
    CopyToVec(constraint_ub_, foot_contact_states, num_legs_, planning_horizon_,
              5, &a_ub);

    // Reduce problem size to make solving faster
    std::vector<int> var_elim(qp_dim, 0);
    std::vector<int> con_elim(constraint_dim, 0);
    int new_vars = qp_dim;
    int new_cons = constraint_dim;

    for (int i = 0; i < constraint_dim; i++) {
      if (near_zero(a_lb[i]) && near_zero(a_ub[i])) {
        for (int j = 2; j < qp_dim; j += 3) {
          if (near_one(a_mat[i * qp_dim + j])) {
            new_vars -= 3;
            new_cons -= 5;
            var_elim[j - 2] = 1;
            var_elim[j - 1] = 1;
            var_elim[j] = 1;
            int cs = (j * 5) / 3 - 3;
            con_elim[cs] = 1;
            con_elim[cs + 1] = 1;
            con_elim[cs + 2] = 1;
            con_elim[cs + 3] = 1;
            con_elim[cs + 4] = 1;
          }
        }
      }
    }
    
    // std::cout << "a_lb" << a_lb[4] << "\n";
    // std::cout << "a_ub" << a_ub[4] << "\n";
    // std::printf("Original vars: %d, cons: %d\n", qp_dim, constraint_dim);
    // std::printf("Reduced vars: %d, cons: %d\n", new_vars, new_cons);

    // Construct reduced qp matrices;
    std::vector<int> var_ind(new_vars, 0);
    std::vector<int> con_ind(new_cons, 0);
    int var_count = 0;
    for (int i = 0; i < qp_dim; i++) {
      if (!var_elim[i]) {
        var_ind[var_count] = i;
        var_count++;
      }
    }

    int con_count = 0;
    for (int i = 0; i < constraint_dim; i++) {
      if (!con_elim[i]) {
        con_ind[con_count] = i;
        con_count++;
      }
    }

    std::vector<qpOASES::real_t> red_hessian(new_vars * new_vars, 0);
    std::vector<qpOASES::real_t> red_g_vec(new_vars, 0);
    std::vector<qpOASES::real_t> red_a_mat(new_cons * new_cons, 0);
    std::vector<qpOASES::real_t> red_a_lb(new_cons, 0);
    std::vector<qpOASES::real_t> red_a_ub(new_cons, 0);

    for (int i = 0; i < new_vars; i++) {
      int oldi = var_ind[i];
      red_g_vec[i] = g_vec[oldi];
      for (int j = 0; j < new_vars; j++) {
        int oldj = var_ind[j];
        red_hessian[i * new_vars + j] = hessian[oldi * qp_dim + oldj];
      }
    }

    for (int i = 0; i < new_cons; i++) {
      int oldi = con_ind[i];
      red_a_lb[i] = a_lb[oldi];
      red_a_ub[i] = a_ub[oldi];
      for (int j = 0; j < new_vars; j++) {
        int oldj = var_ind[j];
        red_a_mat[i * new_vars + j] = a_mat[oldi * qp_dim + oldj];
      }
    }

    auto qp_problem = QProblem(new_vars, new_cons, qpOASES::HST_UNKNOWN,
                              qpOASES::BT_TRUE);

    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    // options.enableEqualities = qpOASES::BT_TRUE;
    qp_problem.setOptions(options);

    qpOASES::int_t max_solver_iter = 100;

    qp_problem.init(red_hessian.data(), red_g_vec.data(), red_a_mat.data(),
                    nullptr, nullptr, red_a_lb.data(), red_a_ub.data(),
                    max_solver_iter);
    std::vector<qpOASES::real_t> qp_red_sol(new_vars);
    qp_problem.getPrimalSolution(qp_red_sol.data());

    // Map the reduced problem back to original problem.
    std::vector<qpOASES::real_t> qp_sol(qp_dim, 0);
    int vc = 0;
    for (int i = 0; i < qp_dim; i++) {
      if (var_elim[i]) {
        qp_sol[i] = 0.0f;
      } else {
        qp_sol[i] = qp_red_sol[vc];
        vc++;
      }
    }
    // auto qp_problem = QProblem(qp_dim, constraint_dim, qpOASES::HST_UNKNOWN,
    //                            qpOASES::BT_TRUE);

    // qpOASES::Options options;
    // options.setToMPC();
    // options.printLevel = qpOASES::PL_NONE;
    // qp_problem.setOptions(options);

    // int max_solver_iter = 100;

    // qp_problem.init(hessian.data(), g_vec.data(), a_mat.data(), nullptr,
    //                 nullptr, a_lb.data(), a_ub.data(), max_solver_iter,
    //                 nullptr);

    // std::vector<qpOASES::real_t> qp_sol(qp_dim, 0);
    // qp_problem.getPrimalSolution(qp_sol.data());

    int buffer_index = 0;
    for (int i = 0; i < num_legs_ * planning_horizon_; i++) {
      // if (foot_contact_states(i / num_legs_, i % num_legs_) == 0) {
      //   qp_solution_[i * k3Dim] = 0;
      //   qp_solution_[i * k3Dim + 1] = 0;
      //   qp_solution_[i * k3Dim + 2] = 0;
      // } else {
      qp_solution_[i * k3Dim] = qp_sol[buffer_index * k3Dim];
      qp_solution_[i * k3Dim + 1] = qp_sol[buffer_index * k3Dim + 1];
      qp_solution_[i * k3Dim + 2] = qp_sol[buffer_index * k3Dim + 2];
      ++buffer_index;
      // }
    }
    // for(int i = 0; i < 12; i++)
    // {
    //   mpc_data << state_[i] << " ";
    // }
    // for(int i = 0; i < 12; i++)
    // {
    //   mpc_data << desired_states_[i] << " ";
    // }
    // for(int i = 0; i < 12; i++)
    // {
    //   mpc_data << qp_solution_[i] << " ";
    // }
    // mpc_data << std::endl;
    return qp_solution_;
}


namespace py = pybind11;

PYBIND11_MODULE(mpc_osqp, m) {
  m.doc() = R"pbdoc(
        MPC using OSQP Python Bindings
        -----------------------

        .. currentmodule:: mpc_osqp

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  py::enum_<QPSolverName>(m, "QPSolverName")
      .value("OSQP", OSQP, "OSQP")
      .value("QPOASES", QPOASES, "QPOASES")
      .export_values();

  py::class_<ConvexMpc>(m, "ConvexMpc")
      .def(py::init<double, const std::vector<double>&, int, int, double,
                    double, QPSolverName>())
      .def("compute_contact_forces", &ConvexMpc::ComputeContactForces)
      .def("reset_solver", &ConvexMpc::ResetSolver);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  m.attr("TEST") = py::int_(int(42));
}