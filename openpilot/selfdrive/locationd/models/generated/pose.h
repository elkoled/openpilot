#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_7924553503606272438);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1094996591856183194);
void pose_H_mod_fun(double *state, double *out_5263462732536222298);
void pose_f_fun(double *state, double dt, double *out_5332214133538366933);
void pose_F_fun(double *state, double dt, double *out_4992300284098693487);
void pose_h_4(double *state, double *unused, double *out_4840607086056089263);
void pose_H_4(double *state, double *unused, double *out_7083161458442519159);
void pose_h_10(double *state, double *unused, double *out_4068132626889395165);
void pose_H_10(double *state, double *unused, double *out_8672064356962455792);
void pose_h_13(double *state, double *unused, double *out_7526180503868782014);
void pose_H_13(double *state, double *unused, double *out_527469749874181770);
void pose_h_14(double *state, double *unused, double *out_2452990038371875929);
void pose_H_14(double *state, double *unused, double *out_3119920602103034630);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}