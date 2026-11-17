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
void car_err_fun(double *nom_x, double *delta_x, double *out_3882781232910498265);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_862445192448987119);
void car_H_mod_fun(double *state, double *out_5987638020737947726);
void car_f_fun(double *state, double dt, double *out_2235320704754806331);
void car_F_fun(double *state, double dt, double *out_3931128456043697150);
void car_h_25(double *state, double *unused, double *out_1729639954696850456);
void car_H_25(double *state, double *unused, double *out_8093305575896703421);
void car_h_24(double *state, double *unused, double *out_4364495401834085915);
void car_H_24(double *state, double *unused, double *out_4535379977094380285);
void car_h_30(double *state, double *unused, double *out_6151632696057198729);
void car_H_30(double *state, double *unused, double *out_1176615234405086666);
void car_h_26(double *state, double *unused, double *out_6810027346311520432);
void car_H_26(double *state, double *unused, double *out_6611935178938791971);
void car_h_27(double *state, double *unused, double *out_3272033312963995189);
void car_H_27(double *state, double *unused, double *out_3351378546205511577);
void car_h_29(double *state, double *unused, double *out_3547227375248501078);
void car_H_29(double *state, double *unused, double *out_666383890090694482);
void car_h_28(double *state, double *unused, double *out_8273706580624840834);
void car_H_28(double *state, double *unused, double *out_5748782907160225056);
void car_h_31(double *state, double *unused, double *out_5367089482950872806);
void car_H_31(double *state, double *unused, double *out_8062659614019742993);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}