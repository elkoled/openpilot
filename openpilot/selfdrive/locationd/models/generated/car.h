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
void car_err_fun(double *nom_x, double *delta_x, double *out_4716796072493757619);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4656456150899037945);
void car_H_mod_fun(double *state, double *out_5303764549756169270);
void car_f_fun(double *state, double dt, double *out_9109159330846486809);
void car_F_fun(double *state, double dt, double *out_2940363433241261210);
void car_h_25(double *state, double *unused, double *out_929019453456471170);
void car_H_25(double *state, double *unused, double *out_7633596646979824642);
void car_h_24(double *state, double *unused, double *out_3208442796249610617);
void car_H_24(double *state, double *unused, double *out_1432563967128663216);
void car_h_30(double *state, double *unused, double *out_6755773258966896457);
void car_H_30(double *state, double *unused, double *out_8294814468222478347);
void car_h_26(double *state, double *unused, double *out_4151367938158198806);
void car_H_26(double *state, double *unused, double *out_3892093328105768418);
void car_h_27(double *state, double *unused, double *out_5930692721117316815);
void car_H_27(double *state, double *unused, double *out_5329494388036159661);
void car_h_29(double *state, double *unused, double *out_1832389422837944951);
void car_H_29(double *state, double *unused, double *out_7784583123908086163);
void car_h_28(double *state, double *unused, double *out_325132000841730439);
void car_H_28(double *state, double *unused, double *out_5579761932731934879);
void car_h_31(double *state, double *unused, double *out_8692284710277143611);
void car_H_31(double *state, double *unused, double *out_7664242608856785070);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}