#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_4845004285734235912) {
   out_4845004285734235912[0] = delta_x[0] + nom_x[0];
   out_4845004285734235912[1] = delta_x[1] + nom_x[1];
   out_4845004285734235912[2] = delta_x[2] + nom_x[2];
   out_4845004285734235912[3] = delta_x[3] + nom_x[3];
   out_4845004285734235912[4] = delta_x[4] + nom_x[4];
   out_4845004285734235912[5] = delta_x[5] + nom_x[5];
   out_4845004285734235912[6] = delta_x[6] + nom_x[6];
   out_4845004285734235912[7] = delta_x[7] + nom_x[7];
   out_4845004285734235912[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7709784370528721654) {
   out_7709784370528721654[0] = -nom_x[0] + true_x[0];
   out_7709784370528721654[1] = -nom_x[1] + true_x[1];
   out_7709784370528721654[2] = -nom_x[2] + true_x[2];
   out_7709784370528721654[3] = -nom_x[3] + true_x[3];
   out_7709784370528721654[4] = -nom_x[4] + true_x[4];
   out_7709784370528721654[5] = -nom_x[5] + true_x[5];
   out_7709784370528721654[6] = -nom_x[6] + true_x[6];
   out_7709784370528721654[7] = -nom_x[7] + true_x[7];
   out_7709784370528721654[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4629646266179334341) {
   out_4629646266179334341[0] = 1.0;
   out_4629646266179334341[1] = 0.0;
   out_4629646266179334341[2] = 0.0;
   out_4629646266179334341[3] = 0.0;
   out_4629646266179334341[4] = 0.0;
   out_4629646266179334341[5] = 0.0;
   out_4629646266179334341[6] = 0.0;
   out_4629646266179334341[7] = 0.0;
   out_4629646266179334341[8] = 0.0;
   out_4629646266179334341[9] = 0.0;
   out_4629646266179334341[10] = 1.0;
   out_4629646266179334341[11] = 0.0;
   out_4629646266179334341[12] = 0.0;
   out_4629646266179334341[13] = 0.0;
   out_4629646266179334341[14] = 0.0;
   out_4629646266179334341[15] = 0.0;
   out_4629646266179334341[16] = 0.0;
   out_4629646266179334341[17] = 0.0;
   out_4629646266179334341[18] = 0.0;
   out_4629646266179334341[19] = 0.0;
   out_4629646266179334341[20] = 1.0;
   out_4629646266179334341[21] = 0.0;
   out_4629646266179334341[22] = 0.0;
   out_4629646266179334341[23] = 0.0;
   out_4629646266179334341[24] = 0.0;
   out_4629646266179334341[25] = 0.0;
   out_4629646266179334341[26] = 0.0;
   out_4629646266179334341[27] = 0.0;
   out_4629646266179334341[28] = 0.0;
   out_4629646266179334341[29] = 0.0;
   out_4629646266179334341[30] = 1.0;
   out_4629646266179334341[31] = 0.0;
   out_4629646266179334341[32] = 0.0;
   out_4629646266179334341[33] = 0.0;
   out_4629646266179334341[34] = 0.0;
   out_4629646266179334341[35] = 0.0;
   out_4629646266179334341[36] = 0.0;
   out_4629646266179334341[37] = 0.0;
   out_4629646266179334341[38] = 0.0;
   out_4629646266179334341[39] = 0.0;
   out_4629646266179334341[40] = 1.0;
   out_4629646266179334341[41] = 0.0;
   out_4629646266179334341[42] = 0.0;
   out_4629646266179334341[43] = 0.0;
   out_4629646266179334341[44] = 0.0;
   out_4629646266179334341[45] = 0.0;
   out_4629646266179334341[46] = 0.0;
   out_4629646266179334341[47] = 0.0;
   out_4629646266179334341[48] = 0.0;
   out_4629646266179334341[49] = 0.0;
   out_4629646266179334341[50] = 1.0;
   out_4629646266179334341[51] = 0.0;
   out_4629646266179334341[52] = 0.0;
   out_4629646266179334341[53] = 0.0;
   out_4629646266179334341[54] = 0.0;
   out_4629646266179334341[55] = 0.0;
   out_4629646266179334341[56] = 0.0;
   out_4629646266179334341[57] = 0.0;
   out_4629646266179334341[58] = 0.0;
   out_4629646266179334341[59] = 0.0;
   out_4629646266179334341[60] = 1.0;
   out_4629646266179334341[61] = 0.0;
   out_4629646266179334341[62] = 0.0;
   out_4629646266179334341[63] = 0.0;
   out_4629646266179334341[64] = 0.0;
   out_4629646266179334341[65] = 0.0;
   out_4629646266179334341[66] = 0.0;
   out_4629646266179334341[67] = 0.0;
   out_4629646266179334341[68] = 0.0;
   out_4629646266179334341[69] = 0.0;
   out_4629646266179334341[70] = 1.0;
   out_4629646266179334341[71] = 0.0;
   out_4629646266179334341[72] = 0.0;
   out_4629646266179334341[73] = 0.0;
   out_4629646266179334341[74] = 0.0;
   out_4629646266179334341[75] = 0.0;
   out_4629646266179334341[76] = 0.0;
   out_4629646266179334341[77] = 0.0;
   out_4629646266179334341[78] = 0.0;
   out_4629646266179334341[79] = 0.0;
   out_4629646266179334341[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_2820358476098001832) {
   out_2820358476098001832[0] = state[0];
   out_2820358476098001832[1] = state[1];
   out_2820358476098001832[2] = state[2];
   out_2820358476098001832[3] = state[3];
   out_2820358476098001832[4] = state[4];
   out_2820358476098001832[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_2820358476098001832[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_2820358476098001832[7] = state[7];
   out_2820358476098001832[8] = state[8];
}
void F_fun(double *state, double dt, double *out_4772620070161941839) {
   out_4772620070161941839[0] = 1;
   out_4772620070161941839[1] = 0;
   out_4772620070161941839[2] = 0;
   out_4772620070161941839[3] = 0;
   out_4772620070161941839[4] = 0;
   out_4772620070161941839[5] = 0;
   out_4772620070161941839[6] = 0;
   out_4772620070161941839[7] = 0;
   out_4772620070161941839[8] = 0;
   out_4772620070161941839[9] = 0;
   out_4772620070161941839[10] = 1;
   out_4772620070161941839[11] = 0;
   out_4772620070161941839[12] = 0;
   out_4772620070161941839[13] = 0;
   out_4772620070161941839[14] = 0;
   out_4772620070161941839[15] = 0;
   out_4772620070161941839[16] = 0;
   out_4772620070161941839[17] = 0;
   out_4772620070161941839[18] = 0;
   out_4772620070161941839[19] = 0;
   out_4772620070161941839[20] = 1;
   out_4772620070161941839[21] = 0;
   out_4772620070161941839[22] = 0;
   out_4772620070161941839[23] = 0;
   out_4772620070161941839[24] = 0;
   out_4772620070161941839[25] = 0;
   out_4772620070161941839[26] = 0;
   out_4772620070161941839[27] = 0;
   out_4772620070161941839[28] = 0;
   out_4772620070161941839[29] = 0;
   out_4772620070161941839[30] = 1;
   out_4772620070161941839[31] = 0;
   out_4772620070161941839[32] = 0;
   out_4772620070161941839[33] = 0;
   out_4772620070161941839[34] = 0;
   out_4772620070161941839[35] = 0;
   out_4772620070161941839[36] = 0;
   out_4772620070161941839[37] = 0;
   out_4772620070161941839[38] = 0;
   out_4772620070161941839[39] = 0;
   out_4772620070161941839[40] = 1;
   out_4772620070161941839[41] = 0;
   out_4772620070161941839[42] = 0;
   out_4772620070161941839[43] = 0;
   out_4772620070161941839[44] = 0;
   out_4772620070161941839[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_4772620070161941839[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_4772620070161941839[47] = -dt*stiffness_front*state[0]/state[1];
   out_4772620070161941839[48] = -dt*stiffness_front*state[0]/state[1];
   out_4772620070161941839[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_4772620070161941839[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_4772620070161941839[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_4772620070161941839[52] = dt*stiffness_front*state[0]/state[1];
   out_4772620070161941839[53] = -9.8100000000000005*dt;
   out_4772620070161941839[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_4772620070161941839[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_4772620070161941839[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4772620070161941839[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4772620070161941839[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_4772620070161941839[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_4772620070161941839[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_4772620070161941839[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_4772620070161941839[62] = 0;
   out_4772620070161941839[63] = 0;
   out_4772620070161941839[64] = 0;
   out_4772620070161941839[65] = 0;
   out_4772620070161941839[66] = 0;
   out_4772620070161941839[67] = 0;
   out_4772620070161941839[68] = 0;
   out_4772620070161941839[69] = 0;
   out_4772620070161941839[70] = 1;
   out_4772620070161941839[71] = 0;
   out_4772620070161941839[72] = 0;
   out_4772620070161941839[73] = 0;
   out_4772620070161941839[74] = 0;
   out_4772620070161941839[75] = 0;
   out_4772620070161941839[76] = 0;
   out_4772620070161941839[77] = 0;
   out_4772620070161941839[78] = 0;
   out_4772620070161941839[79] = 0;
   out_4772620070161941839[80] = 1;
}
void h_25(double *state, double *unused, double *out_7168939626174010290) {
   out_7168939626174010290[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6776805203456714753) {
   out_6776805203456714753[0] = 0;
   out_6776805203456714753[1] = 0;
   out_6776805203456714753[2] = 0;
   out_6776805203456714753[3] = 0;
   out_6776805203456714753[4] = 0;
   out_6776805203456714753[5] = 0;
   out_6776805203456714753[6] = 1;
   out_6776805203456714753[7] = 0;
   out_6776805203456714753[8] = 0;
}
void h_24(double *state, double *unused, double *out_416659608716010498) {
   out_416659608716010498[0] = state[4];
   out_416659608716010498[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6849617365596848600) {
   out_6849617365596848600[0] = 0;
   out_6849617365596848600[1] = 0;
   out_6849617365596848600[2] = 0;
   out_6849617365596848600[3] = 0;
   out_6849617365596848600[4] = 1;
   out_6849617365596848600[5] = 0;
   out_6849617365596848600[6] = 0;
   out_6849617365596848600[7] = 0;
   out_6849617365596848600[8] = 0;
   out_6849617365596848600[9] = 0;
   out_6849617365596848600[10] = 0;
   out_6849617365596848600[11] = 0;
   out_6849617365596848600[12] = 0;
   out_6849617365596848600[13] = 0;
   out_6849617365596848600[14] = 1;
   out_6849617365596848600[15] = 0;
   out_6849617365596848600[16] = 0;
   out_6849617365596848600[17] = 0;
}
void h_30(double *state, double *unused, double *out_7444133688458516179) {
   out_7444133688458516179[0] = state[4];
}
void H_30(double *state, double *unused, double *out_6906144150599954823) {
   out_6906144150599954823[0] = 0;
   out_6906144150599954823[1] = 0;
   out_6906144150599954823[2] = 0;
   out_6906144150599954823[3] = 0;
   out_6906144150599954823[4] = 1;
   out_6906144150599954823[5] = 0;
   out_6906144150599954823[6] = 0;
   out_6906144150599954823[7] = 0;
   out_6906144150599954823[8] = 0;
}
void h_26(double *state, double *unused, double *out_1565848200899801091) {
   out_1565848200899801091[0] = state[7];
}
void H_26(double *state, double *unused, double *out_6119951139346402849) {
   out_6119951139346402849[0] = 0;
   out_6119951139346402849[1] = 0;
   out_6119951139346402849[2] = 0;
   out_6119951139346402849[3] = 0;
   out_6119951139346402849[4] = 0;
   out_6119951139346402849[5] = 0;
   out_6119951139346402849[6] = 0;
   out_6119951139346402849[7] = 1;
   out_6119951139346402849[8] = 0;
}
void h_27(double *state, double *unused, double *out_5478491791438329330) {
   out_5478491791438329330[0] = state[3];
}
void H_27(double *state, double *unused, double *out_9080907462400379734) {
   out_9080907462400379734[0] = 0;
   out_9080907462400379734[1] = 0;
   out_9080907462400379734[2] = 0;
   out_9080907462400379734[3] = 1;
   out_9080907462400379734[4] = 0;
   out_9080907462400379734[5] = 0;
   out_9080907462400379734[6] = 0;
   out_9080907462400379734[7] = 0;
   out_9080907462400379734[8] = 0;
}
void h_29(double *state, double *unused, double *out_6812805086430562157) {
   out_6812805086430562157[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6395912806285562639) {
   out_6395912806285562639[0] = 0;
   out_6395912806285562639[1] = 1;
   out_6395912806285562639[2] = 0;
   out_6395912806285562639[3] = 0;
   out_6395912806285562639[4] = 0;
   out_6395912806285562639[5] = 0;
   out_6395912806285562639[6] = 0;
   out_6395912806285562639[7] = 0;
   out_6395912806285562639[8] = 0;
}
void h_28(double *state, double *unused, double *out_7966579014610376641) {
   out_7966579014610376641[0] = state[0];
}
void H_28(double *state, double *unused, double *out_6968432250354458403) {
   out_6968432250354458403[0] = 1;
   out_6968432250354458403[1] = 0;
   out_6968432250354458403[2] = 0;
   out_6968432250354458403[3] = 0;
   out_6968432250354458403[4] = 0;
   out_6968432250354458403[5] = 0;
   out_6968432250354458403[6] = 0;
   out_6968432250354458403[7] = 0;
   out_6968432250354458403[8] = 0;
}
void h_31(double *state, double *unused, double *out_3384864952651017967) {
   out_3384864952651017967[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6746159241579754325) {
   out_6746159241579754325[0] = 0;
   out_6746159241579754325[1] = 0;
   out_6746159241579754325[2] = 0;
   out_6746159241579754325[3] = 0;
   out_6746159241579754325[4] = 0;
   out_6746159241579754325[5] = 0;
   out_6746159241579754325[6] = 0;
   out_6746159241579754325[7] = 0;
   out_6746159241579754325[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_4845004285734235912) {
  err_fun(nom_x, delta_x, out_4845004285734235912);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7709784370528721654) {
  inv_err_fun(nom_x, true_x, out_7709784370528721654);
}
void car_H_mod_fun(double *state, double *out_4629646266179334341) {
  H_mod_fun(state, out_4629646266179334341);
}
void car_f_fun(double *state, double dt, double *out_2820358476098001832) {
  f_fun(state,  dt, out_2820358476098001832);
}
void car_F_fun(double *state, double dt, double *out_4772620070161941839) {
  F_fun(state,  dt, out_4772620070161941839);
}
void car_h_25(double *state, double *unused, double *out_7168939626174010290) {
  h_25(state, unused, out_7168939626174010290);
}
void car_H_25(double *state, double *unused, double *out_6776805203456714753) {
  H_25(state, unused, out_6776805203456714753);
}
void car_h_24(double *state, double *unused, double *out_416659608716010498) {
  h_24(state, unused, out_416659608716010498);
}
void car_H_24(double *state, double *unused, double *out_6849617365596848600) {
  H_24(state, unused, out_6849617365596848600);
}
void car_h_30(double *state, double *unused, double *out_7444133688458516179) {
  h_30(state, unused, out_7444133688458516179);
}
void car_H_30(double *state, double *unused, double *out_6906144150599954823) {
  H_30(state, unused, out_6906144150599954823);
}
void car_h_26(double *state, double *unused, double *out_1565848200899801091) {
  h_26(state, unused, out_1565848200899801091);
}
void car_H_26(double *state, double *unused, double *out_6119951139346402849) {
  H_26(state, unused, out_6119951139346402849);
}
void car_h_27(double *state, double *unused, double *out_5478491791438329330) {
  h_27(state, unused, out_5478491791438329330);
}
void car_H_27(double *state, double *unused, double *out_9080907462400379734) {
  H_27(state, unused, out_9080907462400379734);
}
void car_h_29(double *state, double *unused, double *out_6812805086430562157) {
  h_29(state, unused, out_6812805086430562157);
}
void car_H_29(double *state, double *unused, double *out_6395912806285562639) {
  H_29(state, unused, out_6395912806285562639);
}
void car_h_28(double *state, double *unused, double *out_7966579014610376641) {
  h_28(state, unused, out_7966579014610376641);
}
void car_H_28(double *state, double *unused, double *out_6968432250354458403) {
  H_28(state, unused, out_6968432250354458403);
}
void car_h_31(double *state, double *unused, double *out_3384864952651017967) {
  h_31(state, unused, out_3384864952651017967);
}
void car_H_31(double *state, double *unused, double *out_6746159241579754325) {
  H_31(state, unused, out_6746159241579754325);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
