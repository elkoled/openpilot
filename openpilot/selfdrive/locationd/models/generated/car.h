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
void car_err_fun(double *nom_x, double *delta_x, double *out_4845004285734235912);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_7709784370528721654);
void car_H_mod_fun(double *state, double *out_4629646266179334341);
void car_f_fun(double *state, double dt, double *out_2820358476098001832);
void car_F_fun(double *state, double dt, double *out_4772620070161941839);
void car_h_25(double *state, double *unused, double *out_7168939626174010290);
void car_H_25(double *state, double *unused, double *out_6776805203456714753);
void car_h_24(double *state, double *unused, double *out_416659608716010498);
void car_H_24(double *state, double *unused, double *out_6849617365596848600);
void car_h_30(double *state, double *unused, double *out_7444133688458516179);
void car_H_30(double *state, double *unused, double *out_6906144150599954823);
void car_h_26(double *state, double *unused, double *out_1565848200899801091);
void car_H_26(double *state, double *unused, double *out_6119951139346402849);
void car_h_27(double *state, double *unused, double *out_5478491791438329330);
void car_H_27(double *state, double *unused, double *out_9080907462400379734);
void car_h_29(double *state, double *unused, double *out_6812805086430562157);
void car_H_29(double *state, double *unused, double *out_6395912806285562639);
void car_h_28(double *state, double *unused, double *out_7966579014610376641);
void car_H_28(double *state, double *unused, double *out_6968432250354458403);
void car_h_31(double *state, double *unused, double *out_3384864952651017967);
void car_H_31(double *state, double *unused, double *out_6746159241579754325);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}