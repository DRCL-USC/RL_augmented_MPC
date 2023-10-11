#include "MPCSolver.h"
#include "utils.h"

//#define K_DEBUG
//#define K_PRINT_EVERYTHING

MPCSolver::MPCSolver(double dt, int horizon, double mu, double f_max)
{
  state_num = 13;
  control_num = 12;
  first_run = true;
  has_solved = 0;
  full_weight.setZero();
  linear_accel.setZero();
  angular_accel.setZero();
  
  setup_problem(dt, horizon, mu, f_max);
}

void MPCSolver::initialize_mpc()
{
  // printf("Initializing MPC!\n");
  if (pthread_mutex_init(&problem_cfg_mt, NULL) != 0)
    printf("[MPC ERROR] Failed to initialize problem configuration mutex.\n");

  if (pthread_mutex_init(&update_mt, NULL) != 0)
    printf("[MPC ERROR] Failed to initialize update data mutex.\n");

#ifdef K_DEBUG
  printf("[MPC] Debugging enabled.\n");
  printf("[MPC] Size of problem setup struct: %ld bytes.\n", sizeof(problem_setup));
  printf("      Size of problem update struct: %ld bytes.\n", sizeof(update_data_t));
  printf("      Size of MATLAB floating point type: %ld bytes.\n", sizeof(mfp));
  printf("      Size of flt: %ld bytes.\n", sizeof(flt));
#else
    // printf("[MPC] Debugging disabled.\n");
#endif
}

void MPCSolver::setup_problem(double dt, int horizon, double mu, double f_max)
{
  if (first_run)
  {
    first_run = false;
    initialize_mpc();
  }

#ifdef K_DEBUG
  printf("[MPC] Got new problem configuration!\n");
  printf("[MPC] Prediction horizon length: %d\n      Force limit: %.3f, friction %.3f\n      dt: %.3f\n",
         horizon, f_max, mu, dt);
#endif

  // pthread_mutex_lock(&problem_cfg_mt);

  setup.horizon = horizon;
  setup.f_max = f_max;
  setup.mu = mu;
  setup.dt = dt;

  // pthread_mutex_unlock(&problem_cfg_mt);
  resize_qp_mats(horizon);
}

void MPCSolver::set_acclerations(Vec3<double> _ang_accel, Vec3<double> _lin_accel)
{
  for(int i = 0; i < 3; i++)
  {
    angular_accel(i) = _ang_accel(i);
    linear_accel(i) = _lin_accel(i);
  }
}

void MPCSolver::update_problem_data(double *p, double *v, double *q, double *w, double *r, double yaw, double *weights,
                                    double *state_trajectory, double alpha, int *gait, int robot_type)
{
  mfp_to_flt(update.p, p, 3);
  mfp_to_flt(update.v, v, 3);
  mfp_to_flt(update.q, q, 4);
  mfp_to_flt(update.w, w, 3);
  mfp_to_flt(update.r, r, 12);
  update.yaw = yaw;
  update.robot_type = robot_type;
  mfp_to_flt(update.weights, weights, 12);
  // this is safe, the solver isn't running, and update_problem_data and setup_problem
  // are called from the same thread
  mfp_to_flt(update.traj, state_trajectory, 12 * setup.horizon);
  update.alpha = alpha;
  mint_to_u8(update.gait, gait, 4 * setup.horizon);
  solve_mpc();
  has_solved = 1;
}

double MPCSolver::get_solution(int index)
{
  if (!has_solved)
    return 0.f;
  return q_soln[index];
}

void MPCSolver::resize_qp_mats(s16 horizon)
{
  int mcount = 0;
  int h2 = horizon * horizon;

  A_qp.resize(state_num * horizon, state_num);
  mcount += state_num * horizon * 1;

  B_qp.resize(state_num * horizon, control_num * horizon);
  mcount += state_num * h2 * control_num;

  S.resize(state_num * horizon, state_num * horizon);
  mcount += state_num * state_num * h2;

  X_d.resize(state_num * horizon, Eigen::NoChange);
  mcount += state_num * horizon;

  U_b.resize(20 * horizon, Eigen::NoChange);
  mcount += 20 * horizon;

  fmat.resize(20 * horizon, control_num * horizon);
  mcount += 20 * control_num * h2;

  qH.resize(control_num * horizon, control_num * horizon);
  mcount += control_num * control_num * h2;

  qg.resize(control_num * horizon, Eigen::NoChange);
  mcount += control_num * horizon;

  eye_12h.resize(control_num * horizon, control_num * horizon);
  mcount += control_num * control_num * horizon;

  Adt.resize(state_num, state_num);
  Bdt.resize(state_num, control_num);
  ABc.resize(state_num + control_num, state_num + control_num);
  expmm.resize(state_num + control_num, state_num + control_num);
  x_0.resize(state_num, 1);
  A.resize(state_num, state_num);
  B.resize(state_num, control_num);

  for (int i = 0; i < 20; i++)
  {
    powerMats[i].resize(state_num, state_num);
  }

  full_weight.resize(state_num, 1);
  // printf("realloc'd %d floating point numbers.\n",mcount);
  mcount = 0;

  A_qp.setZero();
  B_qp.setZero();
  S.setZero();
  X_d.setZero();
  U_b.setZero();
  fmat.setZero();
  qH.setZero();
  eye_12h.setIdentity();

  // TODO: use realloc instead of free/malloc on size changes

  if (real_allocated)
  {

    free(H_qpoases);
    free(g_qpoases);
    free(A_qpoases);
    free(lb_qpoases);
    free(ub_qpoases);
    free(q_soln);
    free(H_red);
    free(g_red);
    free(A_red);
    free(lb_red);
    free(ub_red);
    free(q_red);
  }

  H_qpoases = (qpOASES::real_t *)malloc(control_num * control_num * horizon * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * control_num * h2;
  g_qpoases = (qpOASES::real_t *)malloc(control_num * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * horizon;
  A_qpoases = (qpOASES::real_t *)malloc(control_num * 20 * horizon * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * 20 * h2;
  lb_qpoases = (qpOASES::real_t *)malloc(20 * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += 20 * horizon;
  ub_qpoases = (qpOASES::real_t *)malloc(20 * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += 20 * horizon;
  q_soln = (qpOASES::real_t *)malloc(control_num * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * horizon;

  H_red = (qpOASES::real_t *)malloc(control_num * control_num * horizon * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * control_num * h2;
  g_red = (qpOASES::real_t *)malloc(control_num * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * horizon;
  A_red = (qpOASES::real_t *)malloc(control_num * 20 * horizon * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * 20 * h2;
  lb_red = (qpOASES::real_t *)malloc(20 * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += 20 * horizon;
  ub_red = (qpOASES::real_t *)malloc(20 * 1 * horizon * sizeof(qpOASES::real_t));
  mcount += 20 * horizon;
  q_red = (qpOASES::real_t *)malloc(control_num * horizon * sizeof(qpOASES::real_t));
  mcount += control_num * horizon;
  real_allocated = 1;

  // printf("malloc'd %d floating point numbers.\n",mcount);

#ifdef K_DEBUG
  printf("RESIZED MATRICES FOR HORIZON: %d\n", horizon);
#endif
}

void MPCSolver::c2qp(Eigen::Matrix<fpt, 3, 3> I_world, fpt m, Eigen::Matrix<fpt, 3, 4> r_feet, Eigen::Matrix<fpt, 3, 3> R_yaw, fpt dt, s16 horizon)
{

  A.setZero();
  A(3, 9) = 1.f;
  A(4, 10) = 1.f;
  A(5, 11) = 1.f;
  A(6, 12) = angular_accel(0);
  A(7, 12) = angular_accel(1);
  A(8, 12) = angular_accel(2);
  A(9, 12) = linear_accel(0);
  A(10, 12) = linear_accel(1);
  A(11, 12) = -9.8f + linear_accel(2);
  A.block(0, 6, 3, 3) = R_yaw.transpose();

  if (state_num == 19)
  { // adaptive control
    A.block(6, 13, 6, 6) = Eigen::Matrix<fpt, 6, 6>::Identity();
  }

  B.setZero();
  Eigen::Matrix<fpt, 3, 3> I_inv = I_world.inverse();

  for (s16 b = 0; b < 4; b++)
  {
    B.block(6, b * 3, 3, 3) = cross_mat(I_inv, r_feet.col(b));
    B.block(9, b * 3, 3, 3) = Eigen::Matrix<fpt, 3, 3>::Identity() / m;
  }

  ABc.setZero();
  ABc.block(0, 0, state_num, state_num) = A;
  ABc.block(0, state_num, state_num, control_num) = B;
  ABc = dt * ABc;
  expmm = ABc.exp();
  Adt = expmm.block(0, 0, state_num, state_num);
  Bdt = expmm.block(0, state_num, state_num, control_num);

#ifdef K_PRINT_EVERYTHING
  cout << "Adt: \n"
       << Adt << "\nBdt:\n"
       << Bdt << endl;
#endif
  if (horizon > 19)
  {
    throw std::runtime_error("horizon is too long!");
  }

  powerMats[0].setIdentity();
  for (int i = 1; i < horizon + 1; i++)
  {
    powerMats[i] = Adt * powerMats[i - 1];
  }

  for (s16 r = 0; r < horizon; r++)
  {
    A_qp.block(state_num * r, 0, state_num, state_num) = powerMats[r + 1]; // Adt.pow(r+1);
    for (s16 c = 0; c < horizon; c++)
    {
      if (r >= c)
      {
        s16 a_num = r - c;
        B_qp.block(state_num * r, control_num * c, state_num, control_num) = powerMats[a_num] /*Adt.pow(a_num)*/ * Bdt;
      }
    }
  }
#ifdef K_PRINT_EVERYTHING
  cout << "AQP:\n"
       << A_qp << "\nBQP:\n"
       << B_qp << endl;
#endif
}

void MPCSolver::solve_mpc()
{
  rs.set(update.p, update.v, update.q, update.w, update.r, update.yaw, update.robot_type);
#ifdef K_PRINT_EVERYTHING

  printf("-----------------\n");
  printf("   PROBLEM DATA  \n");
  printf("-----------------\n");
  print_problem_setup(setup);

  printf("-----------------\n");
  printf("    ROBOT DATA   \n");
  printf("-----------------\n");
  rs.print();
  print_update_data(update, setup.horizon);
#endif

  // roll pitch yaw
  Eigen::Matrix<fpt, 3, 1> rpy;
  quat_to_rpy(rs.q, rpy); //rpy is yaw pitch roll

  // initial state (state representation)
  x_0 << rpy(2), rpy(1), rpy(0), rs.p, rs.w, rs.v, 1.f;
  // cout<<state_num<<endl;
  // original
  // I_world = rs.R_yaw.transpose() * rs.I_body * rs.R_yaw;
  // cout<<rs.R_yaw<<endl;

#ifdef K_PRINT_EVERYTHING
  cout << "Initial state: \n"
       << x_0 << endl;
  cout << "World Inertia: \n"
       << I_world << endl;
  cout << "A CT: \n"
       << A_ct << endl;
  cout << "B CT (simplified): \n"
       << B_ct_r << endl;
#endif
  // QP matrices
  c2qp(rs.I_world, rs.m, rs.r_feet, rs.R_yaw, setup.dt, setup.horizon);

  // weights

  for (u8 i = 0; i < 12; i++)
    full_weight(i) = update.weights[i];

  S.diagonal() = full_weight.replicate(setup.horizon, 1);

  // trajectory
  for (s16 i = 0; i < setup.horizon; i++)
  {
    for (s16 j = 0; j < 12; j++)
      X_d(state_num * i + j, 0) = update.traj[12 * i + j];
  }

  // note - I'm not doing the shifting here.
  s16 k = 0;
  for (s16 i = 0; i < setup.horizon; i++)
  {
    for (s16 j = 0; j < 4; j++)
    {
      U_b(5 * k + 0) = BIG_NUMBER;
      U_b(5 * k + 1) = BIG_NUMBER;
      U_b(5 * k + 2) = BIG_NUMBER;
      U_b(5 * k + 3) = BIG_NUMBER;
      U_b(5 * k + 4) = update.gait[i * 4 + j] * setup.f_max;
      k++;
    }
  }

  mu = 1.f / setup.mu;

  f_block << mu, 0, 1.f,
      -mu, 0, 1.f,
      0, mu, 1.f,
      0, -mu, 1.f,
      0, 0, 1.f;

  for (s16 i = 0; i < setup.horizon * 4; i++)
  {
    fmat.block(i * 5, i * 3, 5, 3) = f_block;
  }

  qH = 2 * (B_qp.transpose() * S * B_qp + update.alpha * eye_12h);
  qg = 2 * B_qp.transpose() * S * (A_qp * x_0 - X_d);

  matrix_to_real(H_qpoases, qH, setup.horizon * control_num, setup.horizon * control_num);
  matrix_to_real(g_qpoases, qg, setup.horizon * control_num, 1);
  matrix_to_real(A_qpoases, fmat, setup.horizon * 20, setup.horizon * control_num);
  matrix_to_real(ub_qpoases, U_b, setup.horizon * 20, 1);

  for (s16 i = 0; i < 20 * setup.horizon; i++)
    lb_qpoases[i] = 0.0f;

  s16 num_constraints = 20 * setup.horizon;
  s16 num_variables = control_num * setup.horizon;

  qpOASES::int_t nWSR = 100;

  int new_vars = num_variables;
  int new_cons = num_constraints;

  for (int i = 0; i < num_constraints; i++)
    con_elim[i] = 0;

  for (int i = 0; i < num_variables; i++)
    var_elim[i] = 0;

  for (int i = 0; i < num_constraints; i++)
  {
    if (!(near_zero(lb_qpoases[i]) && near_zero(ub_qpoases[i])))
      continue;
    double *c_row = &A_qpoases[i * num_variables];
    for (int j = 0; j < num_variables; j++)
    {
      if (near_one(c_row[j]))
      {
        new_vars -= 3;
        new_cons -= 5;
        int cs = (j * 5) / 3 - 3;
        var_elim[j - 2] = 1;
        var_elim[j - 1] = 1;
        var_elim[j] = 1;
        con_elim[cs] = 1;
        con_elim[cs + 1] = 1;
        con_elim[cs + 2] = 1;
        con_elim[cs + 3] = 1;
        con_elim[cs + 4] = 1;
      }
    }
  }
  // if(new_vars != num_variables)
  if (1 == 1)
  {
    int var_ind[new_vars];
    int con_ind[new_cons];
    int vc = 0;
    for (int i = 0; i < num_variables; i++)
    {
      if (!var_elim[i])
      {
        if (!(vc < new_vars))
        {
          printf("BAD ERROR 1\n");
        }
        var_ind[vc] = i;
        vc++;
      }
    }
    vc = 0;
    for (int i = 0; i < num_constraints; i++)
    {
      if (!con_elim[i])
      {
        if (!(vc < new_cons))
        {
          printf("BAD ERROR 1\n");
        }
        con_ind[vc] = i;
        vc++;
      }
    }
    for (int i = 0; i < new_vars; i++)
    {
      int olda = var_ind[i];
      g_red[i] = g_qpoases[olda];
      for (int j = 0; j < new_vars; j++)
      {
        int oldb = var_ind[j];
        H_red[i * new_vars + j] = H_qpoases[olda * num_variables + oldb];
      }
    }

    for (int con = 0; con < new_cons; con++)
    {
      for (int st = 0; st < new_vars; st++)
      {
        float cval = A_qpoases[(num_variables * con_ind[con]) + var_ind[st]];
        A_red[con * new_vars + st] = cval;
      }
    }
    for (int i = 0; i < new_cons; i++)
    {
      int old = con_ind[i];
      ub_red[i] = ub_qpoases[old];
      lb_red[i] = lb_qpoases[old];
    }

    Timer solve_timer;
    qpOASES::QProblem problem_red(new_vars, new_cons);
    qpOASES::Options op;
    op.setToMPC();
    op.printLevel = qpOASES::PL_NONE;
    op.enableEqualities = qpOASES::BT_TRUE;
    problem_red.setOptions(op);
    // int_t nWSR = 50000;

    int rval = problem_red.init(H_red, g_red, A_red, NULL, NULL, lb_red, ub_red, nWSR);
    (void)rval;
    int rval2 = problem_red.getPrimalSolution(q_red);
    if (rval2 != qpOASES::SUCCESSFUL_RETURN)
      printf("failed to solve!\n");

    // printf("solve time: %.3f ms, size %d, %d\n", solve_timer.getMs(), new_vars, new_cons);

    vc = 0;
    for (int i = 0; i < num_variables; i++)
    {
      if (var_elim[i])
      {
        q_soln[i] = 0.0f;
      }
      else
      {
        q_soln[i] = q_red[vc];
        vc++;
      }
    }
  }

#ifdef K_PRINT_EVERYTHING
  // cout<<"fmat:\n"<<fmat<<endl;
#endif
}
