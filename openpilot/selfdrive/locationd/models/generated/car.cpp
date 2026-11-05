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
void err_fun(double *nom_x, double *delta_x, double *out_4716796072493757619) {
   out_4716796072493757619[0] = delta_x[0] + nom_x[0];
   out_4716796072493757619[1] = delta_x[1] + nom_x[1];
   out_4716796072493757619[2] = delta_x[2] + nom_x[2];
   out_4716796072493757619[3] = delta_x[3] + nom_x[3];
   out_4716796072493757619[4] = delta_x[4] + nom_x[4];
   out_4716796072493757619[5] = delta_x[5] + nom_x[5];
   out_4716796072493757619[6] = delta_x[6] + nom_x[6];
   out_4716796072493757619[7] = delta_x[7] + nom_x[7];
   out_4716796072493757619[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4656456150899037945) {
   out_4656456150899037945[0] = -nom_x[0] + true_x[0];
   out_4656456150899037945[1] = -nom_x[1] + true_x[1];
   out_4656456150899037945[2] = -nom_x[2] + true_x[2];
   out_4656456150899037945[3] = -nom_x[3] + true_x[3];
   out_4656456150899037945[4] = -nom_x[4] + true_x[4];
   out_4656456150899037945[5] = -nom_x[5] + true_x[5];
   out_4656456150899037945[6] = -nom_x[6] + true_x[6];
   out_4656456150899037945[7] = -nom_x[7] + true_x[7];
   out_4656456150899037945[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_5303764549756169270) {
   out_5303764549756169270[0] = 1.0;
   out_5303764549756169270[1] = 0.0;
   out_5303764549756169270[2] = 0.0;
   out_5303764549756169270[3] = 0.0;
   out_5303764549756169270[4] = 0.0;
   out_5303764549756169270[5] = 0.0;
   out_5303764549756169270[6] = 0.0;
   out_5303764549756169270[7] = 0.0;
   out_5303764549756169270[8] = 0.0;
   out_5303764549756169270[9] = 0.0;
   out_5303764549756169270[10] = 1.0;
   out_5303764549756169270[11] = 0.0;
   out_5303764549756169270[12] = 0.0;
   out_5303764549756169270[13] = 0.0;
   out_5303764549756169270[14] = 0.0;
   out_5303764549756169270[15] = 0.0;
   out_5303764549756169270[16] = 0.0;
   out_5303764549756169270[17] = 0.0;
   out_5303764549756169270[18] = 0.0;
   out_5303764549756169270[19] = 0.0;
   out_5303764549756169270[20] = 1.0;
   out_5303764549756169270[21] = 0.0;
   out_5303764549756169270[22] = 0.0;
   out_5303764549756169270[23] = 0.0;
   out_5303764549756169270[24] = 0.0;
   out_5303764549756169270[25] = 0.0;
   out_5303764549756169270[26] = 0.0;
   out_5303764549756169270[27] = 0.0;
   out_5303764549756169270[28] = 0.0;
   out_5303764549756169270[29] = 0.0;
   out_5303764549756169270[30] = 1.0;
   out_5303764549756169270[31] = 0.0;
   out_5303764549756169270[32] = 0.0;
   out_5303764549756169270[33] = 0.0;
   out_5303764549756169270[34] = 0.0;
   out_5303764549756169270[35] = 0.0;
   out_5303764549756169270[36] = 0.0;
   out_5303764549756169270[37] = 0.0;
   out_5303764549756169270[38] = 0.0;
   out_5303764549756169270[39] = 0.0;
   out_5303764549756169270[40] = 1.0;
   out_5303764549756169270[41] = 0.0;
   out_5303764549756169270[42] = 0.0;
   out_5303764549756169270[43] = 0.0;
   out_5303764549756169270[44] = 0.0;
   out_5303764549756169270[45] = 0.0;
   out_5303764549756169270[46] = 0.0;
   out_5303764549756169270[47] = 0.0;
   out_5303764549756169270[48] = 0.0;
   out_5303764549756169270[49] = 0.0;
   out_5303764549756169270[50] = 1.0;
   out_5303764549756169270[51] = 0.0;
   out_5303764549756169270[52] = 0.0;
   out_5303764549756169270[53] = 0.0;
   out_5303764549756169270[54] = 0.0;
   out_5303764549756169270[55] = 0.0;
   out_5303764549756169270[56] = 0.0;
   out_5303764549756169270[57] = 0.0;
   out_5303764549756169270[58] = 0.0;
   out_5303764549756169270[59] = 0.0;
   out_5303764549756169270[60] = 1.0;
   out_5303764549756169270[61] = 0.0;
   out_5303764549756169270[62] = 0.0;
   out_5303764549756169270[63] = 0.0;
   out_5303764549756169270[64] = 0.0;
   out_5303764549756169270[65] = 0.0;
   out_5303764549756169270[66] = 0.0;
   out_5303764549756169270[67] = 0.0;
   out_5303764549756169270[68] = 0.0;
   out_5303764549756169270[69] = 0.0;
   out_5303764549756169270[70] = 1.0;
   out_5303764549756169270[71] = 0.0;
   out_5303764549756169270[72] = 0.0;
   out_5303764549756169270[73] = 0.0;
   out_5303764549756169270[74] = 0.0;
   out_5303764549756169270[75] = 0.0;
   out_5303764549756169270[76] = 0.0;
   out_5303764549756169270[77] = 0.0;
   out_5303764549756169270[78] = 0.0;
   out_5303764549756169270[79] = 0.0;
   out_5303764549756169270[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_9109159330846486809) {
   out_9109159330846486809[0] = state[0];
   out_9109159330846486809[1] = state[1];
   out_9109159330846486809[2] = state[2];
   out_9109159330846486809[3] = state[3];
   out_9109159330846486809[4] = state[4];
   out_9109159330846486809[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_9109159330846486809[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_9109159330846486809[7] = state[7];
   out_9109159330846486809[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2940363433241261210) {
   out_2940363433241261210[0] = 1;
   out_2940363433241261210[1] = 0;
   out_2940363433241261210[2] = 0;
   out_2940363433241261210[3] = 0;
   out_2940363433241261210[4] = 0;
   out_2940363433241261210[5] = 0;
   out_2940363433241261210[6] = 0;
   out_2940363433241261210[7] = 0;
   out_2940363433241261210[8] = 0;
   out_2940363433241261210[9] = 0;
   out_2940363433241261210[10] = 1;
   out_2940363433241261210[11] = 0;
   out_2940363433241261210[12] = 0;
   out_2940363433241261210[13] = 0;
   out_2940363433241261210[14] = 0;
   out_2940363433241261210[15] = 0;
   out_2940363433241261210[16] = 0;
   out_2940363433241261210[17] = 0;
   out_2940363433241261210[18] = 0;
   out_2940363433241261210[19] = 0;
   out_2940363433241261210[20] = 1;
   out_2940363433241261210[21] = 0;
   out_2940363433241261210[22] = 0;
   out_2940363433241261210[23] = 0;
   out_2940363433241261210[24] = 0;
   out_2940363433241261210[25] = 0;
   out_2940363433241261210[26] = 0;
   out_2940363433241261210[27] = 0;
   out_2940363433241261210[28] = 0;
   out_2940363433241261210[29] = 0;
   out_2940363433241261210[30] = 1;
   out_2940363433241261210[31] = 0;
   out_2940363433241261210[32] = 0;
   out_2940363433241261210[33] = 0;
   out_2940363433241261210[34] = 0;
   out_2940363433241261210[35] = 0;
   out_2940363433241261210[36] = 0;
   out_2940363433241261210[37] = 0;
   out_2940363433241261210[38] = 0;
   out_2940363433241261210[39] = 0;
   out_2940363433241261210[40] = 1;
   out_2940363433241261210[41] = 0;
   out_2940363433241261210[42] = 0;
   out_2940363433241261210[43] = 0;
   out_2940363433241261210[44] = 0;
   out_2940363433241261210[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_2940363433241261210[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_2940363433241261210[47] = -dt*stiffness_front*state[0]/state[1];
   out_2940363433241261210[48] = -dt*stiffness_front*state[0]/state[1];
   out_2940363433241261210[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_2940363433241261210[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_2940363433241261210[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_2940363433241261210[52] = dt*stiffness_front*state[0]/state[1];
   out_2940363433241261210[53] = -9.8100000000000005*dt;
   out_2940363433241261210[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2940363433241261210[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2940363433241261210[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2940363433241261210[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2940363433241261210[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2940363433241261210[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2940363433241261210[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2940363433241261210[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2940363433241261210[62] = 0;
   out_2940363433241261210[63] = 0;
   out_2940363433241261210[64] = 0;
   out_2940363433241261210[65] = 0;
   out_2940363433241261210[66] = 0;
   out_2940363433241261210[67] = 0;
   out_2940363433241261210[68] = 0;
   out_2940363433241261210[69] = 0;
   out_2940363433241261210[70] = 1;
   out_2940363433241261210[71] = 0;
   out_2940363433241261210[72] = 0;
   out_2940363433241261210[73] = 0;
   out_2940363433241261210[74] = 0;
   out_2940363433241261210[75] = 0;
   out_2940363433241261210[76] = 0;
   out_2940363433241261210[77] = 0;
   out_2940363433241261210[78] = 0;
   out_2940363433241261210[79] = 0;
   out_2940363433241261210[80] = 1;
}
void h_25(double *state, double *unused, double *out_929019453456471170) {
   out_929019453456471170[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7633596646979824642) {
   out_7633596646979824642[0] = 0;
   out_7633596646979824642[1] = 0;
   out_7633596646979824642[2] = 0;
   out_7633596646979824642[3] = 0;
   out_7633596646979824642[4] = 0;
   out_7633596646979824642[5] = 0;
   out_7633596646979824642[6] = 1;
   out_7633596646979824642[7] = 0;
   out_7633596646979824642[8] = 0;
}
void h_24(double *state, double *unused, double *out_3208442796249610617) {
   out_3208442796249610617[0] = state[4];
   out_3208442796249610617[1] = state[5];
}
void H_24(double *state, double *unused, double *out_1432563967128663216) {
   out_1432563967128663216[0] = 0;
   out_1432563967128663216[1] = 0;
   out_1432563967128663216[2] = 0;
   out_1432563967128663216[3] = 0;
   out_1432563967128663216[4] = 1;
   out_1432563967128663216[5] = 0;
   out_1432563967128663216[6] = 0;
   out_1432563967128663216[7] = 0;
   out_1432563967128663216[8] = 0;
   out_1432563967128663216[9] = 0;
   out_1432563967128663216[10] = 0;
   out_1432563967128663216[11] = 0;
   out_1432563967128663216[12] = 0;
   out_1432563967128663216[13] = 0;
   out_1432563967128663216[14] = 1;
   out_1432563967128663216[15] = 0;
   out_1432563967128663216[16] = 0;
   out_1432563967128663216[17] = 0;
}
void h_30(double *state, double *unused, double *out_6755773258966896457) {
   out_6755773258966896457[0] = state[4];
}
void H_30(double *state, double *unused, double *out_8294814468222478347) {
   out_8294814468222478347[0] = 0;
   out_8294814468222478347[1] = 0;
   out_8294814468222478347[2] = 0;
   out_8294814468222478347[3] = 0;
   out_8294814468222478347[4] = 1;
   out_8294814468222478347[5] = 0;
   out_8294814468222478347[6] = 0;
   out_8294814468222478347[7] = 0;
   out_8294814468222478347[8] = 0;
}
void h_26(double *state, double *unused, double *out_4151367938158198806) {
   out_4151367938158198806[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3892093328105768418) {
   out_3892093328105768418[0] = 0;
   out_3892093328105768418[1] = 0;
   out_3892093328105768418[2] = 0;
   out_3892093328105768418[3] = 0;
   out_3892093328105768418[4] = 0;
   out_3892093328105768418[5] = 0;
   out_3892093328105768418[6] = 0;
   out_3892093328105768418[7] = 1;
   out_3892093328105768418[8] = 0;
}
void h_27(double *state, double *unused, double *out_5930692721117316815) {
   out_5930692721117316815[0] = state[3];
}
void H_27(double *state, double *unused, double *out_5329494388036159661) {
   out_5329494388036159661[0] = 0;
   out_5329494388036159661[1] = 0;
   out_5329494388036159661[2] = 0;
   out_5329494388036159661[3] = 1;
   out_5329494388036159661[4] = 0;
   out_5329494388036159661[5] = 0;
   out_5329494388036159661[6] = 0;
   out_5329494388036159661[7] = 0;
   out_5329494388036159661[8] = 0;
}
void h_29(double *state, double *unused, double *out_1832389422837944951) {
   out_1832389422837944951[0] = state[1];
}
void H_29(double *state, double *unused, double *out_7784583123908086163) {
   out_7784583123908086163[0] = 0;
   out_7784583123908086163[1] = 1;
   out_7784583123908086163[2] = 0;
   out_7784583123908086163[3] = 0;
   out_7784583123908086163[4] = 0;
   out_7784583123908086163[5] = 0;
   out_7784583123908086163[6] = 0;
   out_7784583123908086163[7] = 0;
   out_7784583123908086163[8] = 0;
}
void h_28(double *state, double *unused, double *out_325132000841730439) {
   out_325132000841730439[0] = state[0];
}
void H_28(double *state, double *unused, double *out_5579761932731934879) {
   out_5579761932731934879[0] = 1;
   out_5579761932731934879[1] = 0;
   out_5579761932731934879[2] = 0;
   out_5579761932731934879[3] = 0;
   out_5579761932731934879[4] = 0;
   out_5579761932731934879[5] = 0;
   out_5579761932731934879[6] = 0;
   out_5579761932731934879[7] = 0;
   out_5579761932731934879[8] = 0;
}
void h_31(double *state, double *unused, double *out_8692284710277143611) {
   out_8692284710277143611[0] = state[8];
}
void H_31(double *state, double *unused, double *out_7664242608856785070) {
   out_7664242608856785070[0] = 0;
   out_7664242608856785070[1] = 0;
   out_7664242608856785070[2] = 0;
   out_7664242608856785070[3] = 0;
   out_7664242608856785070[4] = 0;
   out_7664242608856785070[5] = 0;
   out_7664242608856785070[6] = 0;
   out_7664242608856785070[7] = 0;
   out_7664242608856785070[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_4716796072493757619) {
  err_fun(nom_x, delta_x, out_4716796072493757619);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4656456150899037945) {
  inv_err_fun(nom_x, true_x, out_4656456150899037945);
}
void car_H_mod_fun(double *state, double *out_5303764549756169270) {
  H_mod_fun(state, out_5303764549756169270);
}
void car_f_fun(double *state, double dt, double *out_9109159330846486809) {
  f_fun(state,  dt, out_9109159330846486809);
}
void car_F_fun(double *state, double dt, double *out_2940363433241261210) {
  F_fun(state,  dt, out_2940363433241261210);
}
void car_h_25(double *state, double *unused, double *out_929019453456471170) {
  h_25(state, unused, out_929019453456471170);
}
void car_H_25(double *state, double *unused, double *out_7633596646979824642) {
  H_25(state, unused, out_7633596646979824642);
}
void car_h_24(double *state, double *unused, double *out_3208442796249610617) {
  h_24(state, unused, out_3208442796249610617);
}
void car_H_24(double *state, double *unused, double *out_1432563967128663216) {
  H_24(state, unused, out_1432563967128663216);
}
void car_h_30(double *state, double *unused, double *out_6755773258966896457) {
  h_30(state, unused, out_6755773258966896457);
}
void car_H_30(double *state, double *unused, double *out_8294814468222478347) {
  H_30(state, unused, out_8294814468222478347);
}
void car_h_26(double *state, double *unused, double *out_4151367938158198806) {
  h_26(state, unused, out_4151367938158198806);
}
void car_H_26(double *state, double *unused, double *out_3892093328105768418) {
  H_26(state, unused, out_3892093328105768418);
}
void car_h_27(double *state, double *unused, double *out_5930692721117316815) {
  h_27(state, unused, out_5930692721117316815);
}
void car_H_27(double *state, double *unused, double *out_5329494388036159661) {
  H_27(state, unused, out_5329494388036159661);
}
void car_h_29(double *state, double *unused, double *out_1832389422837944951) {
  h_29(state, unused, out_1832389422837944951);
}
void car_H_29(double *state, double *unused, double *out_7784583123908086163) {
  H_29(state, unused, out_7784583123908086163);
}
void car_h_28(double *state, double *unused, double *out_325132000841730439) {
  h_28(state, unused, out_325132000841730439);
}
void car_H_28(double *state, double *unused, double *out_5579761932731934879) {
  H_28(state, unused, out_5579761932731934879);
}
void car_h_31(double *state, double *unused, double *out_8692284710277143611) {
  h_31(state, unused, out_8692284710277143611);
}
void car_H_31(double *state, double *unused, double *out_7664242608856785070) {
  H_31(state, unused, out_7664242608856785070);
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
