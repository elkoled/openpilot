#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_4215104848141748955);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8045931564468893177);
void pose_H_mod_fun(double *state, double *out_6658005824480305271);
void pose_f_fun(double *state, double dt, double *out_7993529346017171902);
void pose_F_fun(double *state, double dt, double *out_348990674142262734);
void pose_h_4(double *state, double *unused, double *out_6484339586194182173);
void pose_H_4(double *state, double *unused, double *out_2286355841577516071);
void pose_h_10(double *state, double *unused, double *out_2411329706835363596);
void pose_H_10(double *state, double *unused, double *out_4215648645313872337);
void pose_h_13(double *state, double *unused, double *out_7156475162968172826);
void pose_H_13(double *state, double *unused, double *out_925917983754816730);
void pose_h_14(double *state, double *unused, double *out_7515135358449386116);
void pose_H_14(double *state, double *unused, double *out_5369144273872888367);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}