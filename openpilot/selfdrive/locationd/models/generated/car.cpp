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
void err_fun(double *nom_x, double *delta_x, double *out_38713660167380430) {
   out_38713660167380430[0] = delta_x[0] + nom_x[0];
   out_38713660167380430[1] = delta_x[1] + nom_x[1];
   out_38713660167380430[2] = delta_x[2] + nom_x[2];
   out_38713660167380430[3] = delta_x[3] + nom_x[3];
   out_38713660167380430[4] = delta_x[4] + nom_x[4];
   out_38713660167380430[5] = delta_x[5] + nom_x[5];
   out_38713660167380430[6] = delta_x[6] + nom_x[6];
   out_38713660167380430[7] = delta_x[7] + nom_x[7];
   out_38713660167380430[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5596931528852376010) {
   out_5596931528852376010[0] = -nom_x[0] + true_x[0];
   out_5596931528852376010[1] = -nom_x[1] + true_x[1];
   out_5596931528852376010[2] = -nom_x[2] + true_x[2];
   out_5596931528852376010[3] = -nom_x[3] + true_x[3];
   out_5596931528852376010[4] = -nom_x[4] + true_x[4];
   out_5596931528852376010[5] = -nom_x[5] + true_x[5];
   out_5596931528852376010[6] = -nom_x[6] + true_x[6];
   out_5596931528852376010[7] = -nom_x[7] + true_x[7];
   out_5596931528852376010[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_8137722922324331647) {
   out_8137722922324331647[0] = 1.0;
   out_8137722922324331647[1] = 0.0;
   out_8137722922324331647[2] = 0.0;
   out_8137722922324331647[3] = 0.0;
   out_8137722922324331647[4] = 0.0;
   out_8137722922324331647[5] = 0.0;
   out_8137722922324331647[6] = 0.0;
   out_8137722922324331647[7] = 0.0;
   out_8137722922324331647[8] = 0.0;
   out_8137722922324331647[9] = 0.0;
   out_8137722922324331647[10] = 1.0;
   out_8137722922324331647[11] = 0.0;
   out_8137722922324331647[12] = 0.0;
   out_8137722922324331647[13] = 0.0;
   out_8137722922324331647[14] = 0.0;
   out_8137722922324331647[15] = 0.0;
   out_8137722922324331647[16] = 0.0;
   out_8137722922324331647[17] = 0.0;
   out_8137722922324331647[18] = 0.0;
   out_8137722922324331647[19] = 0.0;
   out_8137722922324331647[20] = 1.0;
   out_8137722922324331647[21] = 0.0;
   out_8137722922324331647[22] = 0.0;
   out_8137722922324331647[23] = 0.0;
   out_8137722922324331647[24] = 0.0;
   out_8137722922324331647[25] = 0.0;
   out_8137722922324331647[26] = 0.0;
   out_8137722922324331647[27] = 0.0;
   out_8137722922324331647[28] = 0.0;
   out_8137722922324331647[29] = 0.0;
   out_8137722922324331647[30] = 1.0;
   out_8137722922324331647[31] = 0.0;
   out_8137722922324331647[32] = 0.0;
   out_8137722922324331647[33] = 0.0;
   out_8137722922324331647[34] = 0.0;
   out_8137722922324331647[35] = 0.0;
   out_8137722922324331647[36] = 0.0;
   out_8137722922324331647[37] = 0.0;
   out_8137722922324331647[38] = 0.0;
   out_8137722922324331647[39] = 0.0;
   out_8137722922324331647[40] = 1.0;
   out_8137722922324331647[41] = 0.0;
   out_8137722922324331647[42] = 0.0;
   out_8137722922324331647[43] = 0.0;
   out_8137722922324331647[44] = 0.0;
   out_8137722922324331647[45] = 0.0;
   out_8137722922324331647[46] = 0.0;
   out_8137722922324331647[47] = 0.0;
   out_8137722922324331647[48] = 0.0;
   out_8137722922324331647[49] = 0.0;
   out_8137722922324331647[50] = 1.0;
   out_8137722922324331647[51] = 0.0;
   out_8137722922324331647[52] = 0.0;
   out_8137722922324331647[53] = 0.0;
   out_8137722922324331647[54] = 0.0;
   out_8137722922324331647[55] = 0.0;
   out_8137722922324331647[56] = 0.0;
   out_8137722922324331647[57] = 0.0;
   out_8137722922324331647[58] = 0.0;
   out_8137722922324331647[59] = 0.0;
   out_8137722922324331647[60] = 1.0;
   out_8137722922324331647[61] = 0.0;
   out_8137722922324331647[62] = 0.0;
   out_8137722922324331647[63] = 0.0;
   out_8137722922324331647[64] = 0.0;
   out_8137722922324331647[65] = 0.0;
   out_8137722922324331647[66] = 0.0;
   out_8137722922324331647[67] = 0.0;
   out_8137722922324331647[68] = 0.0;
   out_8137722922324331647[69] = 0.0;
   out_8137722922324331647[70] = 1.0;
   out_8137722922324331647[71] = 0.0;
   out_8137722922324331647[72] = 0.0;
   out_8137722922324331647[73] = 0.0;
   out_8137722922324331647[74] = 0.0;
   out_8137722922324331647[75] = 0.0;
   out_8137722922324331647[76] = 0.0;
   out_8137722922324331647[77] = 0.0;
   out_8137722922324331647[78] = 0.0;
   out_8137722922324331647[79] = 0.0;
   out_8137722922324331647[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_6144720723464928856) {
   out_6144720723464928856[0] = state[0];
   out_6144720723464928856[1] = state[1];
   out_6144720723464928856[2] = state[2];
   out_6144720723464928856[3] = state[3];
   out_6144720723464928856[4] = state[4];
   out_6144720723464928856[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_6144720723464928856[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_6144720723464928856[7] = state[7];
   out_6144720723464928856[8] = state[8];
}
void F_fun(double *state, double dt, double *out_8076075350376008773) {
   out_8076075350376008773[0] = 1;
   out_8076075350376008773[1] = 0;
   out_8076075350376008773[2] = 0;
   out_8076075350376008773[3] = 0;
   out_8076075350376008773[4] = 0;
   out_8076075350376008773[5] = 0;
   out_8076075350376008773[6] = 0;
   out_8076075350376008773[7] = 0;
   out_8076075350376008773[8] = 0;
   out_8076075350376008773[9] = 0;
   out_8076075350376008773[10] = 1;
   out_8076075350376008773[11] = 0;
   out_8076075350376008773[12] = 0;
   out_8076075350376008773[13] = 0;
   out_8076075350376008773[14] = 0;
   out_8076075350376008773[15] = 0;
   out_8076075350376008773[16] = 0;
   out_8076075350376008773[17] = 0;
   out_8076075350376008773[18] = 0;
   out_8076075350376008773[19] = 0;
   out_8076075350376008773[20] = 1;
   out_8076075350376008773[21] = 0;
   out_8076075350376008773[22] = 0;
   out_8076075350376008773[23] = 0;
   out_8076075350376008773[24] = 0;
   out_8076075350376008773[25] = 0;
   out_8076075350376008773[26] = 0;
   out_8076075350376008773[27] = 0;
   out_8076075350376008773[28] = 0;
   out_8076075350376008773[29] = 0;
   out_8076075350376008773[30] = 1;
   out_8076075350376008773[31] = 0;
   out_8076075350376008773[32] = 0;
   out_8076075350376008773[33] = 0;
   out_8076075350376008773[34] = 0;
   out_8076075350376008773[35] = 0;
   out_8076075350376008773[36] = 0;
   out_8076075350376008773[37] = 0;
   out_8076075350376008773[38] = 0;
   out_8076075350376008773[39] = 0;
   out_8076075350376008773[40] = 1;
   out_8076075350376008773[41] = 0;
   out_8076075350376008773[42] = 0;
   out_8076075350376008773[43] = 0;
   out_8076075350376008773[44] = 0;
   out_8076075350376008773[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_8076075350376008773[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_8076075350376008773[47] = -dt*stiffness_front*state[0]/state[1];
   out_8076075350376008773[48] = -dt*stiffness_front*state[0]/state[1];
   out_8076075350376008773[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_8076075350376008773[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_8076075350376008773[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_8076075350376008773[52] = dt*stiffness_front*state[0]/state[1];
   out_8076075350376008773[53] = -9.8100000000000005*dt;
   out_8076075350376008773[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_8076075350376008773[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_8076075350376008773[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8076075350376008773[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8076075350376008773[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_8076075350376008773[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_8076075350376008773[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_8076075350376008773[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_8076075350376008773[62] = 0;
   out_8076075350376008773[63] = 0;
   out_8076075350376008773[64] = 0;
   out_8076075350376008773[65] = 0;
   out_8076075350376008773[66] = 0;
   out_8076075350376008773[67] = 0;
   out_8076075350376008773[68] = 0;
   out_8076075350376008773[69] = 0;
   out_8076075350376008773[70] = 1;
   out_8076075350376008773[71] = 0;
   out_8076075350376008773[72] = 0;
   out_8076075350376008773[73] = 0;
   out_8076075350376008773[74] = 0;
   out_8076075350376008773[75] = 0;
   out_8076075350376008773[76] = 0;
   out_8076075350376008773[77] = 0;
   out_8076075350376008773[78] = 0;
   out_8076075350376008773[79] = 0;
   out_8076075350376008773[80] = 1;
}
void h_25(double *state, double *unused, double *out_1195422111234018188) {
   out_1195422111234018188[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1055465303587905590) {
   out_1055465303587905590[0] = 0;
   out_1055465303587905590[1] = 0;
   out_1055465303587905590[2] = 0;
   out_1055465303587905590[3] = 0;
   out_1055465303587905590[4] = 0;
   out_1055465303587905590[5] = 0;
   out_1055465303587905590[6] = 1;
   out_1055465303587905590[7] = 0;
   out_1055465303587905590[8] = 0;
}
void h_24(double *state, double *unused, double *out_2061832432988921346) {
   out_2061832432988921346[0] = state[4];
   out_2061832432988921346[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1170242480390962972) {
   out_1170242480390962972[0] = 0;
   out_1170242480390962972[1] = 0;
   out_1170242480390962972[2] = 0;
   out_1170242480390962972[3] = 0;
   out_1170242480390962972[4] = 1;
   out_1170242480390962972[5] = 0;
   out_1170242480390962972[6] = 0;
   out_1170242480390962972[7] = 0;
   out_1170242480390962972[8] = 0;
   out_1170242480390962972[9] = 0;
   out_1170242480390962972[10] = 0;
   out_1170242480390962972[11] = 0;
   out_1170242480390962972[12] = 0;
   out_1170242480390962972[13] = 0;
   out_1170242480390962972[14] = 1;
   out_1170242480390962972[15] = 0;
   out_1170242480390962972[16] = 0;
   out_1170242480390962972[17] = 0;
}
void h_30(double *state, double *unused, double *out_8958687368054690629) {
   out_8958687368054690629[0] = state[4];
}
void H_30(double *state, double *unused, double *out_5861225037903711165) {
   out_5861225037903711165[0] = 0;
   out_5861225037903711165[1] = 0;
   out_5861225037903711165[2] = 0;
   out_5861225037903711165[3] = 0;
   out_5861225037903711165[4] = 1;
   out_5861225037903711165[5] = 0;
   out_5861225037903711165[6] = 0;
   out_5861225037903711165[7] = 0;
   out_5861225037903711165[8] = 0;
}
void h_26(double *state, double *unused, double *out_3609771218096145899) {
   out_3609771218096145899[0] = state[7];
}
void H_26(double *state, double *unused, double *out_4796968622461961814) {
   out_4796968622461961814[0] = 0;
   out_4796968622461961814[1] = 0;
   out_4796968622461961814[2] = 0;
   out_4796968622461961814[3] = 0;
   out_4796968622461961814[4] = 0;
   out_4796968622461961814[5] = 0;
   out_4796968622461961814[6] = 0;
   out_4796968622461961814[7] = 1;
   out_4796968622461961814[8] = 0;
}
void h_27(double *state, double *unused, double *out_6608457322561657447) {
   out_6608457322561657447[0] = state[3];
}
void H_27(double *state, double *unused, double *out_3686461726103286254) {
   out_3686461726103286254[0] = 0;
   out_3686461726103286254[1] = 0;
   out_3686461726103286254[2] = 0;
   out_3686461726103286254[3] = 1;
   out_3686461726103286254[4] = 0;
   out_3686461726103286254[5] = 0;
   out_3686461726103286254[6] = 0;
   out_3686461726103286254[7] = 0;
   out_3686461726103286254[8] = 0;
}
void h_29(double *state, double *unused, double *out_3957014100393844984) {
   out_3957014100393844984[0] = state[1];
}
void H_29(double *state, double *unused, double *out_6371456382218103349) {
   out_6371456382218103349[0] = 0;
   out_6371456382218103349[1] = 1;
   out_6371456382218103349[2] = 0;
   out_6371456382218103349[3] = 0;
   out_6371456382218103349[4] = 0;
   out_6371456382218103349[5] = 0;
   out_6371456382218103349[6] = 0;
   out_6371456382218103349[7] = 0;
   out_6371456382218103349[8] = 0;
}
void h_28(double *state, double *unused, double *out_5922655997414031833) {
   out_5922655997414031833[0] = state[0];
}
void H_28(double *state, double *unused, double *out_3109300017835795353) {
   out_3109300017835795353[0] = 1;
   out_3109300017835795353[1] = 0;
   out_3109300017835795353[2] = 0;
   out_3109300017835795353[3] = 0;
   out_3109300017835795353[4] = 0;
   out_3109300017835795353[5] = 0;
   out_3109300017835795353[6] = 0;
   out_3109300017835795353[7] = 0;
   out_3109300017835795353[8] = 0;
}
void h_31(double *state, double *unused, double *out_1340941935454673159) {
   out_1340941935454673159[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1024819341710945162) {
   out_1024819341710945162[0] = 0;
   out_1024819341710945162[1] = 0;
   out_1024819341710945162[2] = 0;
   out_1024819341710945162[3] = 0;
   out_1024819341710945162[4] = 0;
   out_1024819341710945162[5] = 0;
   out_1024819341710945162[6] = 0;
   out_1024819341710945162[7] = 0;
   out_1024819341710945162[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_38713660167380430) {
  err_fun(nom_x, delta_x, out_38713660167380430);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5596931528852376010) {
  inv_err_fun(nom_x, true_x, out_5596931528852376010);
}
void car_H_mod_fun(double *state, double *out_8137722922324331647) {
  H_mod_fun(state, out_8137722922324331647);
}
void car_f_fun(double *state, double dt, double *out_6144720723464928856) {
  f_fun(state,  dt, out_6144720723464928856);
}
void car_F_fun(double *state, double dt, double *out_8076075350376008773) {
  F_fun(state,  dt, out_8076075350376008773);
}
void car_h_25(double *state, double *unused, double *out_1195422111234018188) {
  h_25(state, unused, out_1195422111234018188);
}
void car_H_25(double *state, double *unused, double *out_1055465303587905590) {
  H_25(state, unused, out_1055465303587905590);
}
void car_h_24(double *state, double *unused, double *out_2061832432988921346) {
  h_24(state, unused, out_2061832432988921346);
}
void car_H_24(double *state, double *unused, double *out_1170242480390962972) {
  H_24(state, unused, out_1170242480390962972);
}
void car_h_30(double *state, double *unused, double *out_8958687368054690629) {
  h_30(state, unused, out_8958687368054690629);
}
void car_H_30(double *state, double *unused, double *out_5861225037903711165) {
  H_30(state, unused, out_5861225037903711165);
}
void car_h_26(double *state, double *unused, double *out_3609771218096145899) {
  h_26(state, unused, out_3609771218096145899);
}
void car_H_26(double *state, double *unused, double *out_4796968622461961814) {
  H_26(state, unused, out_4796968622461961814);
}
void car_h_27(double *state, double *unused, double *out_6608457322561657447) {
  h_27(state, unused, out_6608457322561657447);
}
void car_H_27(double *state, double *unused, double *out_3686461726103286254) {
  H_27(state, unused, out_3686461726103286254);
}
void car_h_29(double *state, double *unused, double *out_3957014100393844984) {
  h_29(state, unused, out_3957014100393844984);
}
void car_H_29(double *state, double *unused, double *out_6371456382218103349) {
  H_29(state, unused, out_6371456382218103349);
}
void car_h_28(double *state, double *unused, double *out_5922655997414031833) {
  h_28(state, unused, out_5922655997414031833);
}
void car_H_28(double *state, double *unused, double *out_3109300017835795353) {
  H_28(state, unused, out_3109300017835795353);
}
void car_h_31(double *state, double *unused, double *out_1340941935454673159) {
  h_31(state, unused, out_1340941935454673159);
}
void car_H_31(double *state, double *unused, double *out_1024819341710945162) {
  H_31(state, unused, out_1024819341710945162);
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
