#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_5957606077413925934);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1181409950769297795);
void pose_H_mod_fun(double *state, double *out_1576273921813987430);
void pose_f_fun(double *state, double dt, double *out_8480623208419059652);
void pose_F_fun(double *state, double dt, double *out_6948763257431598150);
void pose_h_4(double *state, double *unused, double *out_7022920791114964455);
void pose_H_4(double *state, double *unused, double *out_2330434923868700037);
void pose_h_10(double *state, double *unused, double *out_8675686222808342194);
void pose_H_10(double *state, double *unused, double *out_638794999717561462);
void pose_h_13(double *state, double *unused, double *out_4089366875480109516);
void pose_H_13(double *state, double *unused, double *out_5542708749201032838);
void pose_h_14(double *state, double *unused, double *out_7178822489175268675);
void pose_H_14(double *state, double *unused, double *out_6293675780208184566);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}