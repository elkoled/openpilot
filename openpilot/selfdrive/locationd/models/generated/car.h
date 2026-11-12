#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_6899444373903234522);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_2265990755906395257);
void car_H_mod_fun(double *state, double *out_1896215134982892478);
void car_f_fun(double *state, double dt, double *out_7412003781362127783);
void car_F_fun(double *state, double dt, double *out_8002518847624218306);
void car_h_25(double *state, double *unused, double *out_5048431871801651264);
void car_H_25(double *state, double *unused, double *out_1390880692782595766);
void car_h_24(double *state, double *unused, double *out_2731233124606523423);
void car_H_24(double *state, double *unused, double *out_5373203844180363664);
void car_h_30(double *state, double *unused, double *out_5190815455824993502);
void car_H_30(double *state, double *unused, double *out_5525809648709020989);
void car_h_26(double *state, double *unused, double *out_7570986828459211851);
void car_H_26(double *state, double *unused, double *out_5132384011656651990);
void car_h_27(double *state, double *unused, double *out_46758604140805619);
void car_H_27(double *state, double *unused, double *out_3351046336908596078);
void car_h_29(double *state, double *unused, double *out_3590690924113216731);
void car_H_29(double *state, double *unused, double *out_6036040993023413173);
void car_h_28(double *state, double *unused, double *out_2651163924949503270);
void car_H_28(double *state, double *unused, double *out_953641975953882599);
void car_h_31(double *state, double *unused, double *out_2844507623082872095);
void car_H_31(double *state, double *unused, double *out_1360234730905635338);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}