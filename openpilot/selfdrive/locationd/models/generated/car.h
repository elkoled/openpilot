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
void car_err_fun(double *nom_x, double *delta_x, double *out_7398279557831877279);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_5970806887533953714);
void car_H_mod_fun(double *state, double *out_2025426774738485256);
void car_f_fun(double *state, double dt, double *out_7986114478197093731);
void car_F_fun(double *state, double dt, double *out_2512004386184477462);
void car_h_25(double *state, double *unused, double *out_7245815125992169774);
void car_H_25(double *state, double *unused, double *out_6929118599923150096);
void car_h_24(double *state, double *unused, double *out_1766086214804849581);
void car_H_24(double *state, double *unused, double *out_6532473295569290601);
void car_h_30(double *state, double *unused, double *out_3997712990731395872);
void car_H_30(double *state, double *unused, double *out_4600935132294784765);
void car_h_26(double *state, double *unused, double *out_4366215742898966234);
void car_H_26(double *state, double *unused, double *out_3187615281049093872);
void car_h_27(double *state, double *unused, double *out_999364068531996306);
void car_H_27(double *state, double *unused, double *out_6775698444095209676);
void car_h_29(double *state, double *unused, double *out_1393307669922698221);
void car_H_29(double *state, double *unused, double *out_8489061170964760709);
void car_h_28(double *state, double *unused, double *out_7220633043371386722);
void car_H_28(double *state, double *unused, double *out_4875283885675260333);
void car_h_31(double *state, double *unused, double *out_7521009188276675663);
void car_H_31(double *state, double *unused, double *out_6959764561800110524);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}