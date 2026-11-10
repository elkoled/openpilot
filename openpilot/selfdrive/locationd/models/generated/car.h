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
void car_err_fun(double *nom_x, double *delta_x, double *out_5787383484345050204);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_4301722856823877998);
void car_H_mod_fun(double *state, double *out_3831961659698698883);
void car_f_fun(double *state, double dt, double *out_2176851940149485963);
void car_F_fun(double *state, double dt, double *out_7705248644935014638);
void car_h_25(double *state, double *unused, double *out_8958408446621293744);
void car_H_25(double *state, double *unused, double *out_5850624553951584136);
void car_h_24(double *state, double *unused, double *out_3487693577342896414);
void car_H_24(double *state, double *unused, double *out_2292698955149261000);
void car_h_30(double *state, double *unused, double *out_1595396132203734513);
void car_H_30(double *state, double *unused, double *out_1066065787540032619);
void car_h_26(double *state, double *unused, double *out_4632182183648067198);
void car_H_26(double *state, double *unused, double *out_8854616200883911256);
void car_h_27(double *state, double *unused, double *out_6451549575643925187);
void car_H_27(double *state, double *unused, double *out_1108697524260392292);
void car_h_29(double *state, double *unused, double *out_8055038477239819963);
void car_H_29(double *state, double *unused, double *out_2822060251129943325);
void car_h_28(double *state, double *unused, double *out_5508039722742262752);
void car_H_28(double *state, double *unused, double *out_7904459268199473899);
void car_h_31(double *state, double *unused, double *out_9213141564803751983);
void car_H_31(double *state, double *unused, double *out_5819978592074623708);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}