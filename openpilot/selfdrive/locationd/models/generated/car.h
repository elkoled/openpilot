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
void car_err_fun(double *nom_x, double *delta_x, double *out_38713660167380430);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5596931528852376010);
void car_H_mod_fun(double *state, double *out_8137722922324331647);
void car_f_fun(double *state, double dt, double *out_6144720723464928856);
void car_F_fun(double *state, double dt, double *out_8076075350376008773);
void car_h_25(double *state, double *unused, double *out_1195422111234018188);
void car_H_25(double *state, double *unused, double *out_1055465303587905590);
void car_h_24(double *state, double *unused, double *out_2061832432988921346);
void car_H_24(double *state, double *unused, double *out_1170242480390962972);
void car_h_30(double *state, double *unused, double *out_8958687368054690629);
void car_H_30(double *state, double *unused, double *out_5861225037903711165);
void car_h_26(double *state, double *unused, double *out_3609771218096145899);
void car_H_26(double *state, double *unused, double *out_4796968622461961814);
void car_h_27(double *state, double *unused, double *out_6608457322561657447);
void car_H_27(double *state, double *unused, double *out_3686461726103286254);
void car_h_29(double *state, double *unused, double *out_3957014100393844984);
void car_H_29(double *state, double *unused, double *out_6371456382218103349);
void car_h_28(double *state, double *unused, double *out_5922655997414031833);
void car_H_28(double *state, double *unused, double *out_3109300017835795353);
void car_h_31(double *state, double *unused, double *out_1340941935454673159);
void car_H_31(double *state, double *unused, double *out_1024819341710945162);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}