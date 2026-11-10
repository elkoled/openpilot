#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void pose_err_fun(double *nom_x, double *delta_x, double *out_2016014972386344778);
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_7065715320158728118);
void pose_H_mod_fun(double *state, double *out_515136171011412265);
void pose_f_fun(double *state, double dt, double *out_1656902809266479078);
void pose_F_fun(double *state, double dt, double *out_4521003238642649691);
void pose_h_4(double *state, double *unused, double *out_6145288799047026529);
void pose_H_4(double *state, double *unused, double *out_5332945635740546456);
void pose_h_10(double *state, double *unused, double *out_569963962585618949);
void pose_H_10(double *state, double *unused, double *out_7139448533129631908);
void pose_h_13(double *state, double *unused, double *out_7745490486830301841);
void pose_H_13(double *state, double *unused, double *out_2277685572576154473);
void pose_h_14(double *state, double *unused, double *out_7326154253764583640);
void pose_H_14(double *state, double *unused, double *out_1369704779401061927);
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt);
}