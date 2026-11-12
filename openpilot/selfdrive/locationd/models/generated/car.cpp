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
void err_fun(double *nom_x, double *delta_x, double *out_6899444373903234522) {
   out_6899444373903234522[0] = delta_x[0] + nom_x[0];
   out_6899444373903234522[1] = delta_x[1] + nom_x[1];
   out_6899444373903234522[2] = delta_x[2] + nom_x[2];
   out_6899444373903234522[3] = delta_x[3] + nom_x[3];
   out_6899444373903234522[4] = delta_x[4] + nom_x[4];
   out_6899444373903234522[5] = delta_x[5] + nom_x[5];
   out_6899444373903234522[6] = delta_x[6] + nom_x[6];
   out_6899444373903234522[7] = delta_x[7] + nom_x[7];
   out_6899444373903234522[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_2265990755906395257) {
   out_2265990755906395257[0] = -nom_x[0] + true_x[0];
   out_2265990755906395257[1] = -nom_x[1] + true_x[1];
   out_2265990755906395257[2] = -nom_x[2] + true_x[2];
   out_2265990755906395257[3] = -nom_x[3] + true_x[3];
   out_2265990755906395257[4] = -nom_x[4] + true_x[4];
   out_2265990755906395257[5] = -nom_x[5] + true_x[5];
   out_2265990755906395257[6] = -nom_x[6] + true_x[6];
   out_2265990755906395257[7] = -nom_x[7] + true_x[7];
   out_2265990755906395257[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_1896215134982892478) {
   out_1896215134982892478[0] = 1.0;
   out_1896215134982892478[1] = 0.0;
   out_1896215134982892478[2] = 0.0;
   out_1896215134982892478[3] = 0.0;
   out_1896215134982892478[4] = 0.0;
   out_1896215134982892478[5] = 0.0;
   out_1896215134982892478[6] = 0.0;
   out_1896215134982892478[7] = 0.0;
   out_1896215134982892478[8] = 0.0;
   out_1896215134982892478[9] = 0.0;
   out_1896215134982892478[10] = 1.0;
   out_1896215134982892478[11] = 0.0;
   out_1896215134982892478[12] = 0.0;
   out_1896215134982892478[13] = 0.0;
   out_1896215134982892478[14] = 0.0;
   out_1896215134982892478[15] = 0.0;
   out_1896215134982892478[16] = 0.0;
   out_1896215134982892478[17] = 0.0;
   out_1896215134982892478[18] = 0.0;
   out_1896215134982892478[19] = 0.0;
   out_1896215134982892478[20] = 1.0;
   out_1896215134982892478[21] = 0.0;
   out_1896215134982892478[22] = 0.0;
   out_1896215134982892478[23] = 0.0;
   out_1896215134982892478[24] = 0.0;
   out_1896215134982892478[25] = 0.0;
   out_1896215134982892478[26] = 0.0;
   out_1896215134982892478[27] = 0.0;
   out_1896215134982892478[28] = 0.0;
   out_1896215134982892478[29] = 0.0;
   out_1896215134982892478[30] = 1.0;
   out_1896215134982892478[31] = 0.0;
   out_1896215134982892478[32] = 0.0;
   out_1896215134982892478[33] = 0.0;
   out_1896215134982892478[34] = 0.0;
   out_1896215134982892478[35] = 0.0;
   out_1896215134982892478[36] = 0.0;
   out_1896215134982892478[37] = 0.0;
   out_1896215134982892478[38] = 0.0;
   out_1896215134982892478[39] = 0.0;
   out_1896215134982892478[40] = 1.0;
   out_1896215134982892478[41] = 0.0;
   out_1896215134982892478[42] = 0.0;
   out_1896215134982892478[43] = 0.0;
   out_1896215134982892478[44] = 0.0;
   out_1896215134982892478[45] = 0.0;
   out_1896215134982892478[46] = 0.0;
   out_1896215134982892478[47] = 0.0;
   out_1896215134982892478[48] = 0.0;
   out_1896215134982892478[49] = 0.0;
   out_1896215134982892478[50] = 1.0;
   out_1896215134982892478[51] = 0.0;
   out_1896215134982892478[52] = 0.0;
   out_1896215134982892478[53] = 0.0;
   out_1896215134982892478[54] = 0.0;
   out_1896215134982892478[55] = 0.0;
   out_1896215134982892478[56] = 0.0;
   out_1896215134982892478[57] = 0.0;
   out_1896215134982892478[58] = 0.0;
   out_1896215134982892478[59] = 0.0;
   out_1896215134982892478[60] = 1.0;
   out_1896215134982892478[61] = 0.0;
   out_1896215134982892478[62] = 0.0;
   out_1896215134982892478[63] = 0.0;
   out_1896215134982892478[64] = 0.0;
   out_1896215134982892478[65] = 0.0;
   out_1896215134982892478[66] = 0.0;
   out_1896215134982892478[67] = 0.0;
   out_1896215134982892478[68] = 0.0;
   out_1896215134982892478[69] = 0.0;
   out_1896215134982892478[70] = 1.0;
   out_1896215134982892478[71] = 0.0;
   out_1896215134982892478[72] = 0.0;
   out_1896215134982892478[73] = 0.0;
   out_1896215134982892478[74] = 0.0;
   out_1896215134982892478[75] = 0.0;
   out_1896215134982892478[76] = 0.0;
   out_1896215134982892478[77] = 0.0;
   out_1896215134982892478[78] = 0.0;
   out_1896215134982892478[79] = 0.0;
   out_1896215134982892478[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7412003781362127783) {
   out_7412003781362127783[0] = state[0];
   out_7412003781362127783[1] = state[1];
   out_7412003781362127783[2] = state[2];
   out_7412003781362127783[3] = state[3];
   out_7412003781362127783[4] = state[4];
   out_7412003781362127783[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_7412003781362127783[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7412003781362127783[7] = state[7];
   out_7412003781362127783[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8002518847624218306) {
   out_8002518847624218306[0] = 1;
   out_8002518847624218306[1] = 0;
   out_8002518847624218306[2] = 0;
   out_8002518847624218306[3] = 0;
   out_8002518847624218306[4] = 0;
   out_8002518847624218306[5] = 0;
   out_8002518847624218306[6] = 0;
   out_8002518847624218306[7] = 0;
   out_8002518847624218306[8] = 0;
   out_8002518847624218306[9] = 0;
   out_8002518847624218306[10] = 1;
   out_8002518847624218306[11] = 0;
   out_8002518847624218306[12] = 0;
   out_8002518847624218306[13] = 0;
   out_8002518847624218306[14] = 0;
   out_8002518847624218306[15] = 0;
   out_8002518847624218306[16] = 0;
   out_8002518847624218306[17] = 0;
   out_8002518847624218306[18] = 0;
   out_8002518847624218306[19] = 0;
   out_8002518847624218306[20] = 1;
   out_8002518847624218306[21] = 0;
   out_8002518847624218306[22] = 0;
   out_8002518847624218306[23] = 0;
   out_8002518847624218306[24] = 0;
   out_8002518847624218306[25] = 0;
   out_8002518847624218306[26] = 0;
   out_8002518847624218306[27] = 0;
   out_8002518847624218306[28] = 0;
   out_8002518847624218306[29] = 0;
   out_8002518847624218306[30] = 1;
   out_8002518847624218306[31] = 0;
   out_8002518847624218306[32] = 0;
   out_8002518847624218306[33] = 0;
   out_8002518847624218306[34] = 0;
   out_8002518847624218306[35] = 0;
   out_8002518847624218306[36] = 0;
   out_8002518847624218306[37] = 0;
   out_8002518847624218306[38] = 0;
   out_8002518847624218306[39] = 0;
   out_8002518847624218306[40] = 1;
   out_8002518847624218306[41] = 0;
   out_8002518847624218306[42] = 0;
   out_8002518847624218306[43] = 0;
   out_8002518847624218306[44] = 0;
   out_8002518847624218306[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_8002518847624218306[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_8002518847624218306[47] = -dt*stiffness_front*state[0]/state[1];
   out_8002518847624218306[48] = -dt*stiffness_front*state[0]/state[1];
   out_8002518847624218306[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_8002518847624218306[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_8002518847624218306[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_8002518847624218306[52] = dt*stiffness_front*state[0]/state[1];
   out_8002518847624218306[53] = -9.8100000000000005*dt;
   out_8002518847624218306[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8002518847624218306[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8002518847624218306[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8002518847624218306[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8002518847624218306[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8002518847624218306[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8002518847624218306[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8002518847624218306[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8002518847624218306[62] = 0;
   out_8002518847624218306[63] = 0;
   out_8002518847624218306[64] = 0;
   out_8002518847624218306[65] = 0;
   out_8002518847624218306[66] = 0;
   out_8002518847624218306[67] = 0;
   out_8002518847624218306[68] = 0;
   out_8002518847624218306[69] = 0;
   out_8002518847624218306[70] = 1;
   out_8002518847624218306[71] = 0;
   out_8002518847624218306[72] = 0;
   out_8002518847624218306[73] = 0;
   out_8002518847624218306[74] = 0;
   out_8002518847624218306[75] = 0;
   out_8002518847624218306[76] = 0;
   out_8002518847624218306[77] = 0;
   out_8002518847624218306[78] = 0;
   out_8002518847624218306[79] = 0;
   out_8002518847624218306[80] = 1;
}
void h_25(double *state, double *unused, double *out_5048431871801651264) {
   out_5048431871801651264[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1390880692782595766) {
   out_1390880692782595766[0] = 0;
   out_1390880692782595766[1] = 0;
   out_1390880692782595766[2] = 0;
   out_1390880692782595766[3] = 0;
   out_1390880692782595766[4] = 0;
   out_1390880692782595766[5] = 0;
   out_1390880692782595766[6] = 1;
   out_1390880692782595766[7] = 0;
   out_1390880692782595766[8] = 0;
}
void h_24(double *state, double *unused, double *out_2731233124606523423) {
   out_2731233124606523423[0] = state[4];
   out_2731233124606523423[1] = state[5];
}
void H_24(double *state, double *unused, double *out_5373203844180363664) {
   out_5373203844180363664[0] = 0;
   out_5373203844180363664[1] = 0;
   out_5373203844180363664[2] = 0;
   out_5373203844180363664[3] = 0;
   out_5373203844180363664[4] = 1;
   out_5373203844180363664[5] = 0;
   out_5373203844180363664[6] = 0;
   out_5373203844180363664[7] = 0;
   out_5373203844180363664[8] = 0;
   out_5373203844180363664[9] = 0;
   out_5373203844180363664[10] = 0;
   out_5373203844180363664[11] = 0;
   out_5373203844180363664[12] = 0;
   out_5373203844180363664[13] = 0;
   out_5373203844180363664[14] = 1;
   out_5373203844180363664[15] = 0;
   out_5373203844180363664[16] = 0;
   out_5373203844180363664[17] = 0;
}
void h_30(double *state, double *unused, double *out_5190815455824993502) {
   out_5190815455824993502[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5525809648709020989) {
   out_5525809648709020989[0] = 0;
   out_5525809648709020989[1] = 0;
   out_5525809648709020989[2] = 0;
   out_5525809648709020989[3] = 0;
   out_5525809648709020989[4] = 1;
   out_5525809648709020989[5] = 0;
   out_5525809648709020989[6] = 0;
   out_5525809648709020989[7] = 0;
   out_5525809648709020989[8] = 0;
}
void h_26(double *state, double *unused, double *out_7570986828459211851) {
   out_7570986828459211851[0] = state[7];
}
void H_26(double *state, double *unused, double *out_5132384011656651990) {
   out_5132384011656651990[0] = 0;
   out_5132384011656651990[1] = 0;
   out_5132384011656651990[2] = 0;
   out_5132384011656651990[3] = 0;
   out_5132384011656651990[4] = 0;
   out_5132384011656651990[5] = 0;
   out_5132384011656651990[6] = 0;
   out_5132384011656651990[7] = 1;
   out_5132384011656651990[8] = 0;
}
void h_27(double *state, double *unused, double *out_46758604140805619) {
   out_46758604140805619[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3351046336908596078) {
   out_3351046336908596078[0] = 0;
   out_3351046336908596078[1] = 0;
   out_3351046336908596078[2] = 0;
   out_3351046336908596078[3] = 1;
   out_3351046336908596078[4] = 0;
   out_3351046336908596078[5] = 0;
   out_3351046336908596078[6] = 0;
   out_3351046336908596078[7] = 0;
   out_3351046336908596078[8] = 0;
}
void h_29(double *state, double *unused, double *out_3590690924113216731) {
   out_3590690924113216731[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6036040993023413173) {
   out_6036040993023413173[0] = 0;
   out_6036040993023413173[1] = 1;
   out_6036040993023413173[2] = 0;
   out_6036040993023413173[3] = 0;
   out_6036040993023413173[4] = 0;
   out_6036040993023413173[5] = 0;
   out_6036040993023413173[6] = 0;
   out_6036040993023413173[7] = 0;
   out_6036040993023413173[8] = 0;
}
void h_28(double *state, double *unused, double *out_2651163924949503270) {
   out_2651163924949503270[0] = state[0];
}
void H_28(double *state, double *unused, double *out_953641975953882599) {
   out_953641975953882599[0] = 1;
   out_953641975953882599[1] = 0;
   out_953641975953882599[2] = 0;
   out_953641975953882599[3] = 0;
   out_953641975953882599[4] = 0;
   out_953641975953882599[5] = 0;
   out_953641975953882599[6] = 0;
   out_953641975953882599[7] = 0;
   out_953641975953882599[8] = 0;
}
void h_31(double *state, double *unused, double *out_2844507623082872095) {
   out_2844507623082872095[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1360234730905635338) {
   out_1360234730905635338[0] = 0;
   out_1360234730905635338[1] = 0;
   out_1360234730905635338[2] = 0;
   out_1360234730905635338[3] = 0;
   out_1360234730905635338[4] = 0;
   out_1360234730905635338[5] = 0;
   out_1360234730905635338[6] = 0;
   out_1360234730905635338[7] = 0;
   out_1360234730905635338[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_6899444373903234522) {
  err_fun(nom_x, delta_x, out_6899444373903234522);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2265990755906395257) {
  inv_err_fun(nom_x, true_x, out_2265990755906395257);
}
void car_H_mod_fun(double *state, double *out_1896215134982892478) {
  H_mod_fun(state, out_1896215134982892478);
}
void car_f_fun(double *state, double dt, double *out_7412003781362127783) {
  f_fun(state,  dt, out_7412003781362127783);
}
void car_F_fun(double *state, double dt, double *out_8002518847624218306) {
  F_fun(state,  dt, out_8002518847624218306);
}
void car_h_25(double *state, double *unused, double *out_5048431871801651264) {
  h_25(state, unused, out_5048431871801651264);
}
void car_H_25(double *state, double *unused, double *out_1390880692782595766) {
  H_25(state, unused, out_1390880692782595766);
}
void car_h_24(double *state, double *unused, double *out_2731233124606523423) {
  h_24(state, unused, out_2731233124606523423);
}
void car_H_24(double *state, double *unused, double *out_5373203844180363664) {
  H_24(state, unused, out_5373203844180363664);
}
void car_h_30(double *state, double *unused, double *out_5190815455824993502) {
  h_30(state, unused, out_5190815455824993502);
}
void car_H_30(double *state, double *unused, double *out_5525809648709020989) {
  H_30(state, unused, out_5525809648709020989);
}
void car_h_26(double *state, double *unused, double *out_7570986828459211851) {
  h_26(state, unused, out_7570986828459211851);
}
void car_H_26(double *state, double *unused, double *out_5132384011656651990) {
  H_26(state, unused, out_5132384011656651990);
}
void car_h_27(double *state, double *unused, double *out_46758604140805619) {
  h_27(state, unused, out_46758604140805619);
}
void car_H_27(double *state, double *unused, double *out_3351046336908596078) {
  H_27(state, unused, out_3351046336908596078);
}
void car_h_29(double *state, double *unused, double *out_3590690924113216731) {
  h_29(state, unused, out_3590690924113216731);
}
void car_H_29(double *state, double *unused, double *out_6036040993023413173) {
  H_29(state, unused, out_6036040993023413173);
}
void car_h_28(double *state, double *unused, double *out_2651163924949503270) {
  h_28(state, unused, out_2651163924949503270);
}
void car_H_28(double *state, double *unused, double *out_953641975953882599) {
  H_28(state, unused, out_953641975953882599);
}
void car_h_31(double *state, double *unused, double *out_2844507623082872095) {
  h_31(state, unused, out_2844507623082872095);
}
void car_H_31(double *state, double *unused, double *out_1360234730905635338) {
  H_31(state, unused, out_1360234730905635338);
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
