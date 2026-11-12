#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_8199520216570231398);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6664218853987700651);
void pose_H_mod_fun(double *state, double *out_5334441709381707352);
void pose_f_fun(double *state, double dt, double *out_5577327214882856470);
void pose_F_fun(double *state, double dt, double *out_153840886414794699);
void pose_h_4(double *state, double *unused, double *out_4292041528838637925);
void pose_H_4(double *state, double *unused, double *out_7264220557575885543);
void pose_h_10(double *state, double *unused, double *out_9179668145558624298);
void pose_H_10(double *state, double *unused, double *out_5444952318043486315);
void pose_h_13(double *state, double *unused, double *out_6555536983378449736);
void pose_H_13(double *state, double *unused, double *out_7970249690801333272);
void pose_h_14(double *state, double *unused, double *out_7794980382073332925);
void pose_H_14(double *state, double *unused, double *out_7219282659794181544);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}