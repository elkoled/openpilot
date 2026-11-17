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
void err_fun(double *nom_x, double *delta_x, double *out_7398279557831877279) {
   out_7398279557831877279[0] = delta_x[0] + nom_x[0];
   out_7398279557831877279[1] = delta_x[1] + nom_x[1];
   out_7398279557831877279[2] = delta_x[2] + nom_x[2];
   out_7398279557831877279[3] = delta_x[3] + nom_x[3];
   out_7398279557831877279[4] = delta_x[4] + nom_x[4];
   out_7398279557831877279[5] = delta_x[5] + nom_x[5];
   out_7398279557831877279[6] = delta_x[6] + nom_x[6];
   out_7398279557831877279[7] = delta_x[7] + nom_x[7];
   out_7398279557831877279[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_5970806887533953714) {
   out_5970806887533953714[0] = -nom_x[0] + true_x[0];
   out_5970806887533953714[1] = -nom_x[1] + true_x[1];
   out_5970806887533953714[2] = -nom_x[2] + true_x[2];
   out_5970806887533953714[3] = -nom_x[3] + true_x[3];
   out_5970806887533953714[4] = -nom_x[4] + true_x[4];
   out_5970806887533953714[5] = -nom_x[5] + true_x[5];
   out_5970806887533953714[6] = -nom_x[6] + true_x[6];
   out_5970806887533953714[7] = -nom_x[7] + true_x[7];
   out_5970806887533953714[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_2025426774738485256) {
   out_2025426774738485256[0] = 1.0;
   out_2025426774738485256[1] = 0.0;
   out_2025426774738485256[2] = 0.0;
   out_2025426774738485256[3] = 0.0;
   out_2025426774738485256[4] = 0.0;
   out_2025426774738485256[5] = 0.0;
   out_2025426774738485256[6] = 0.0;
   out_2025426774738485256[7] = 0.0;
   out_2025426774738485256[8] = 0.0;
   out_2025426774738485256[9] = 0.0;
   out_2025426774738485256[10] = 1.0;
   out_2025426774738485256[11] = 0.0;
   out_2025426774738485256[12] = 0.0;
   out_2025426774738485256[13] = 0.0;
   out_2025426774738485256[14] = 0.0;
   out_2025426774738485256[15] = 0.0;
   out_2025426774738485256[16] = 0.0;
   out_2025426774738485256[17] = 0.0;
   out_2025426774738485256[18] = 0.0;
   out_2025426774738485256[19] = 0.0;
   out_2025426774738485256[20] = 1.0;
   out_2025426774738485256[21] = 0.0;
   out_2025426774738485256[22] = 0.0;
   out_2025426774738485256[23] = 0.0;
   out_2025426774738485256[24] = 0.0;
   out_2025426774738485256[25] = 0.0;
   out_2025426774738485256[26] = 0.0;
   out_2025426774738485256[27] = 0.0;
   out_2025426774738485256[28] = 0.0;
   out_2025426774738485256[29] = 0.0;
   out_2025426774738485256[30] = 1.0;
   out_2025426774738485256[31] = 0.0;
   out_2025426774738485256[32] = 0.0;
   out_2025426774738485256[33] = 0.0;
   out_2025426774738485256[34] = 0.0;
   out_2025426774738485256[35] = 0.0;
   out_2025426774738485256[36] = 0.0;
   out_2025426774738485256[37] = 0.0;
   out_2025426774738485256[38] = 0.0;
   out_2025426774738485256[39] = 0.0;
   out_2025426774738485256[40] = 1.0;
   out_2025426774738485256[41] = 0.0;
   out_2025426774738485256[42] = 0.0;
   out_2025426774738485256[43] = 0.0;
   out_2025426774738485256[44] = 0.0;
   out_2025426774738485256[45] = 0.0;
   out_2025426774738485256[46] = 0.0;
   out_2025426774738485256[47] = 0.0;
   out_2025426774738485256[48] = 0.0;
   out_2025426774738485256[49] = 0.0;
   out_2025426774738485256[50] = 1.0;
   out_2025426774738485256[51] = 0.0;
   out_2025426774738485256[52] = 0.0;
   out_2025426774738485256[53] = 0.0;
   out_2025426774738485256[54] = 0.0;
   out_2025426774738485256[55] = 0.0;
   out_2025426774738485256[56] = 0.0;
   out_2025426774738485256[57] = 0.0;
   out_2025426774738485256[58] = 0.0;
   out_2025426774738485256[59] = 0.0;
   out_2025426774738485256[60] = 1.0;
   out_2025426774738485256[61] = 0.0;
   out_2025426774738485256[62] = 0.0;
   out_2025426774738485256[63] = 0.0;
   out_2025426774738485256[64] = 0.0;
   out_2025426774738485256[65] = 0.0;
   out_2025426774738485256[66] = 0.0;
   out_2025426774738485256[67] = 0.0;
   out_2025426774738485256[68] = 0.0;
   out_2025426774738485256[69] = 0.0;
   out_2025426774738485256[70] = 1.0;
   out_2025426774738485256[71] = 0.0;
   out_2025426774738485256[72] = 0.0;
   out_2025426774738485256[73] = 0.0;
   out_2025426774738485256[74] = 0.0;
   out_2025426774738485256[75] = 0.0;
   out_2025426774738485256[76] = 0.0;
   out_2025426774738485256[77] = 0.0;
   out_2025426774738485256[78] = 0.0;
   out_2025426774738485256[79] = 0.0;
   out_2025426774738485256[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_7986114478197093731) {
   out_7986114478197093731[0] = state[0];
   out_7986114478197093731[1] = state[1];
   out_7986114478197093731[2] = state[2];
   out_7986114478197093731[3] = state[3];
   out_7986114478197093731[4] = state[4];
   out_7986114478197093731[5] = dt*(stiffness_front*(-state[2] - state[3] + state[7])*state[0]/state[1] + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/state[4] + ((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4])*state[6] - 9.8100000000000005*state[8]) + state[5];
   out_7986114478197093731[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_7986114478197093731[7] = state[7];
   out_7986114478197093731[8] = state[8];
}
void F_fun(double *state, double dt, double *out_2512004386184477462) {
   out_2512004386184477462[0] = 1;
   out_2512004386184477462[1] = 0;
   out_2512004386184477462[2] = 0;
   out_2512004386184477462[3] = 0;
   out_2512004386184477462[4] = 0;
   out_2512004386184477462[5] = 0;
   out_2512004386184477462[6] = 0;
   out_2512004386184477462[7] = 0;
   out_2512004386184477462[8] = 0;
   out_2512004386184477462[9] = 0;
   out_2512004386184477462[10] = 1;
   out_2512004386184477462[11] = 0;
   out_2512004386184477462[12] = 0;
   out_2512004386184477462[13] = 0;
   out_2512004386184477462[14] = 0;
   out_2512004386184477462[15] = 0;
   out_2512004386184477462[16] = 0;
   out_2512004386184477462[17] = 0;
   out_2512004386184477462[18] = 0;
   out_2512004386184477462[19] = 0;
   out_2512004386184477462[20] = 1;
   out_2512004386184477462[21] = 0;
   out_2512004386184477462[22] = 0;
   out_2512004386184477462[23] = 0;
   out_2512004386184477462[24] = 0;
   out_2512004386184477462[25] = 0;
   out_2512004386184477462[26] = 0;
   out_2512004386184477462[27] = 0;
   out_2512004386184477462[28] = 0;
   out_2512004386184477462[29] = 0;
   out_2512004386184477462[30] = 1;
   out_2512004386184477462[31] = 0;
   out_2512004386184477462[32] = 0;
   out_2512004386184477462[33] = 0;
   out_2512004386184477462[34] = 0;
   out_2512004386184477462[35] = 0;
   out_2512004386184477462[36] = 0;
   out_2512004386184477462[37] = 0;
   out_2512004386184477462[38] = 0;
   out_2512004386184477462[39] = 0;
   out_2512004386184477462[40] = 1;
   out_2512004386184477462[41] = 0;
   out_2512004386184477462[42] = 0;
   out_2512004386184477462[43] = 0;
   out_2512004386184477462[44] = 0;
   out_2512004386184477462[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/state[1] + (-stiffness_front - stiffness_rear)*state[5]/state[4] + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/state[4]);
   out_2512004386184477462[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/pow(state[1], 2);
   out_2512004386184477462[47] = -dt*stiffness_front*state[0]/state[1];
   out_2512004386184477462[48] = -dt*stiffness_front*state[0]/state[1];
   out_2512004386184477462[49] = dt*(-(-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/pow(state[4], 2) + (-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/pow(state[4], 2) - 1)*state[6]);
   out_2512004386184477462[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/state[4] + 1;
   out_2512004386184477462[51] = dt*((-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/state[4] - state[4]);
   out_2512004386184477462[52] = dt*stiffness_front*state[0]/state[1];
   out_2512004386184477462[53] = -9.8100000000000005*dt;
   out_2512004386184477462[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_2512004386184477462[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_2512004386184477462[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2512004386184477462[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2512004386184477462[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_2512004386184477462[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_2512004386184477462[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_2512004386184477462[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_2512004386184477462[62] = 0;
   out_2512004386184477462[63] = 0;
   out_2512004386184477462[64] = 0;
   out_2512004386184477462[65] = 0;
   out_2512004386184477462[66] = 0;
   out_2512004386184477462[67] = 0;
   out_2512004386184477462[68] = 0;
   out_2512004386184477462[69] = 0;
   out_2512004386184477462[70] = 1;
   out_2512004386184477462[71] = 0;
   out_2512004386184477462[72] = 0;
   out_2512004386184477462[73] = 0;
   out_2512004386184477462[74] = 0;
   out_2512004386184477462[75] = 0;
   out_2512004386184477462[76] = 0;
   out_2512004386184477462[77] = 0;
   out_2512004386184477462[78] = 0;
   out_2512004386184477462[79] = 0;
   out_2512004386184477462[80] = 1;
}
void h_25(double *state, double *unused, double *out_7245815125992169774) {
   out_7245815125992169774[0] = state[6];
}
void H_25(double *state, double *unused, double *out_6929118599923150096) {
   out_6929118599923150096[0] = 0;
   out_6929118599923150096[1] = 0;
   out_6929118599923150096[2] = 0;
   out_6929118599923150096[3] = 0;
   out_6929118599923150096[4] = 0;
   out_6929118599923150096[5] = 0;
   out_6929118599923150096[6] = 1;
   out_6929118599923150096[7] = 0;
   out_6929118599923150096[8] = 0;
}
void h_24(double *state, double *unused, double *out_1766086214804849581) {
   out_1766086214804849581[0] = state[4];
   out_1766086214804849581[1] = state[5];
}
void H_24(double *state, double *unused, double *out_6532473295569290601) {
   out_6532473295569290601[0] = 0;
   out_6532473295569290601[1] = 0;
   out_6532473295569290601[2] = 0;
   out_6532473295569290601[3] = 0;
   out_6532473295569290601[4] = 1;
   out_6532473295569290601[5] = 0;
   out_6532473295569290601[6] = 0;
   out_6532473295569290601[7] = 0;
   out_6532473295569290601[8] = 0;
   out_6532473295569290601[9] = 0;
   out_6532473295569290601[10] = 0;
   out_6532473295569290601[11] = 0;
   out_6532473295569290601[12] = 0;
   out_6532473295569290601[13] = 0;
   out_6532473295569290601[14] = 1;
   out_6532473295569290601[15] = 0;
   out_6532473295569290601[16] = 0;
   out_6532473295569290601[17] = 0;
}
void h_30(double *state, double *unused, double *out_3997712990731395872) {
   out_3997712990731395872[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4600935132294784765) {
   out_4600935132294784765[0] = 0;
   out_4600935132294784765[1] = 0;
   out_4600935132294784765[2] = 0;
   out_4600935132294784765[3] = 0;
   out_4600935132294784765[4] = 1;
   out_4600935132294784765[5] = 0;
   out_4600935132294784765[6] = 0;
   out_4600935132294784765[7] = 0;
   out_4600935132294784765[8] = 0;
}
void h_26(double *state, double *unused, double *out_4366215742898966234) {
   out_4366215742898966234[0] = state[7];
}
void H_26(double *state, double *unused, double *out_3187615281049093872) {
   out_3187615281049093872[0] = 0;
   out_3187615281049093872[1] = 0;
   out_3187615281049093872[2] = 0;
   out_3187615281049093872[3] = 0;
   out_3187615281049093872[4] = 0;
   out_3187615281049093872[5] = 0;
   out_3187615281049093872[6] = 0;
   out_3187615281049093872[7] = 1;
   out_3187615281049093872[8] = 0;
}
void h_27(double *state, double *unused, double *out_999364068531996306) {
   out_999364068531996306[0] = state[3];
}
void H_27(double *state, double *unused, double *out_6775698444095209676) {
   out_6775698444095209676[0] = 0;
   out_6775698444095209676[1] = 0;
   out_6775698444095209676[2] = 0;
   out_6775698444095209676[3] = 1;
   out_6775698444095209676[4] = 0;
   out_6775698444095209676[5] = 0;
   out_6775698444095209676[6] = 0;
   out_6775698444095209676[7] = 0;
   out_6775698444095209676[8] = 0;
}
void h_29(double *state, double *unused, double *out_1393307669922698221) {
   out_1393307669922698221[0] = state[1];
}
void H_29(double *state, double *unused, double *out_8489061170964760709) {
   out_8489061170964760709[0] = 0;
   out_8489061170964760709[1] = 1;
   out_8489061170964760709[2] = 0;
   out_8489061170964760709[3] = 0;
   out_8489061170964760709[4] = 0;
   out_8489061170964760709[5] = 0;
   out_8489061170964760709[6] = 0;
   out_8489061170964760709[7] = 0;
   out_8489061170964760709[8] = 0;
}
void h_28(double *state, double *unused, double *out_7220633043371386722) {
   out_7220633043371386722[0] = state[0];
}
void H_28(double *state, double *unused, double *out_4875283885675260333) {
   out_4875283885675260333[0] = 1;
   out_4875283885675260333[1] = 0;
   out_4875283885675260333[2] = 0;
   out_4875283885675260333[3] = 0;
   out_4875283885675260333[4] = 0;
   out_4875283885675260333[5] = 0;
   out_4875283885675260333[6] = 0;
   out_4875283885675260333[7] = 0;
   out_4875283885675260333[8] = 0;
}
void h_31(double *state, double *unused, double *out_7521009188276675663) {
   out_7521009188276675663[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6959764561800110524) {
   out_6959764561800110524[0] = 0;
   out_6959764561800110524[1] = 0;
   out_6959764561800110524[2] = 0;
   out_6959764561800110524[3] = 0;
   out_6959764561800110524[4] = 0;
   out_6959764561800110524[5] = 0;
   out_6959764561800110524[6] = 0;
   out_6959764561800110524[7] = 0;
   out_6959764561800110524[8] = 1;
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
void car_err_fun(double *nom_x, double *delta_x, double *out_7398279557831877279) {
  err_fun(nom_x, delta_x, out_7398279557831877279);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5970806887533953714) {
  inv_err_fun(nom_x, true_x, out_5970806887533953714);
}
void car_H_mod_fun(double *state, double *out_2025426774738485256) {
  H_mod_fun(state, out_2025426774738485256);
}
void car_f_fun(double *state, double dt, double *out_7986114478197093731) {
  f_fun(state,  dt, out_7986114478197093731);
}
void car_F_fun(double *state, double dt, double *out_2512004386184477462) {
  F_fun(state,  dt, out_2512004386184477462);
}
void car_h_25(double *state, double *unused, double *out_7245815125992169774) {
  h_25(state, unused, out_7245815125992169774);
}
void car_H_25(double *state, double *unused, double *out_6929118599923150096) {
  H_25(state, unused, out_6929118599923150096);
}
void car_h_24(double *state, double *unused, double *out_1766086214804849581) {
  h_24(state, unused, out_1766086214804849581);
}
void car_H_24(double *state, double *unused, double *out_6532473295569290601) {
  H_24(state, unused, out_6532473295569290601);
}
void car_h_30(double *state, double *unused, double *out_3997712990731395872) {
  h_30(state, unused, out_3997712990731395872);
}
void car_H_30(double *state, double *unused, double *out_4600935132294784765) {
  H_30(state, unused, out_4600935132294784765);
}
void car_h_26(double *state, double *unused, double *out_4366215742898966234) {
  h_26(state, unused, out_4366215742898966234);
}
void car_H_26(double *state, double *unused, double *out_3187615281049093872) {
  H_26(state, unused, out_3187615281049093872);
}
void car_h_27(double *state, double *unused, double *out_999364068531996306) {
  h_27(state, unused, out_999364068531996306);
}
void car_H_27(double *state, double *unused, double *out_6775698444095209676) {
  H_27(state, unused, out_6775698444095209676);
}
void car_h_29(double *state, double *unused, double *out_1393307669922698221) {
  h_29(state, unused, out_1393307669922698221);
}
void car_H_29(double *state, double *unused, double *out_8489061170964760709) {
  H_29(state, unused, out_8489061170964760709);
}
void car_h_28(double *state, double *unused, double *out_7220633043371386722) {
  h_28(state, unused, out_7220633043371386722);
}
void car_H_28(double *state, double *unused, double *out_4875283885675260333) {
  H_28(state, unused, out_4875283885675260333);
}
void car_h_31(double *state, double *unused, double *out_7521009188276675663) {
  h_31(state, unused, out_7521009188276675663);
}
void car_H_31(double *state, double *unused, double *out_6959764561800110524) {
  H_31(state, unused, out_6959764561800110524);
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
