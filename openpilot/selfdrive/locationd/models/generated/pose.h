#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8643847172382266479);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1016762889049648120);
void pose_H_mod_fun(double *state, double *out_835593229450821917);
void pose_f_fun(double *state, double dt, double *out_3336150215992616844);
void pose_F_fun(double *state, double dt, double *out_8403437658028834064);
void pose_h_4(double *state, double *unused, double *out_4224362500834756736);
void pose_H_4(double *state, double *unused, double *out_4036003130552291941);
void pose_h_10(double *state, double *unused, double *out_7916371707404069928);
void pose_H_10(double *state, double *unused, double *out_2905842532402009938);
void pose_h_13(double *state, double *unused, double *out_1381557700253186812);
void pose_H_13(double *state, double *unused, double *out_3574628077764408988);
void pose_h_14(double *state, double *unused, double *out_7719697066185784283);
void pose_H_14(double *state, double *unused, double *out_7118791562847664237);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}