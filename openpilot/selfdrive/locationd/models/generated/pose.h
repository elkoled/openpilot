#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_5632398071682842920);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_4793870659787952118);
void pose_H_mod_fun(double *state, double *out_4028297677886627584);
void pose_f_fun(double *state, double dt, double *out_2980855877602976982);
void pose_F_fun(double *state, double dt, double *out_2442030683835704114);
void pose_h_4(double *state, double *unused, double *out_5673673386662285868);
void pose_H_4(double *state, double *unused, double *out_256490468042720012);
void pose_h_10(double *state, double *unused, double *out_5579523858078536433);
void pose_H_10(double *state, double *unused, double *out_4722200315799129268);
void pose_h_13(double *state, double *unused, double *out_3238583888242902416);
void pose_H_13(double *state, double *unused, double *out_4090245931345244036);
void pose_h_14(double *state, double *unused, double *out_6143154130072800365);
void pose_H_14(double *state, double *unused, double *out_3339278900338092308);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}