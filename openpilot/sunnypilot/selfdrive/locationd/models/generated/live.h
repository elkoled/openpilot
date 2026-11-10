#pragma once
#include "rednose/helpers/ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_428631491320572367);
void live_err_fun(double *nom_x, double *delta_x, double *out_1486693621419478040);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_5568408697597753011);
void live_H_mod_fun(double *state, double *out_7803690631093836211);
void live_f_fun(double *state, double dt, double *out_2824466101688944296);
void live_F_fun(double *state, double dt, double *out_7929052851099480770);
void live_h_4(double *state, double *unused, double *out_3345104622792572208);
void live_H_4(double *state, double *unused, double *out_1495101082215387743);
void live_h_9(double *state, double *unused, double *out_1008748947527782496);
void live_H_9(double *state, double *unused, double *out_5792117853049059727);
void live_h_10(double *state, double *unused, double *out_3060394618411657729);
void live_H_10(double *state, double *unused, double *out_2914226821293337736);
void live_h_12(double *state, double *unused, double *out_8759323153886029190);
void live_H_12(double *state, double *unused, double *out_3524355325816574052);
void live_h_35(double *state, double *unused, double *out_2625792757130613347);
void live_H_35(double *state, double *unused, double *out_1871560975157219633);
void live_h_32(double *state, double *unused, double *out_7940958264587459135);
void live_H_32(double *state, double *unused, double *out_2347138262514689437);
void live_h_13(double *state, double *unused, double *out_3688638671828090626);
void live_H_13(double *state, double *unused, double *out_5941050363364190705);
void live_h_14(double *state, double *unused, double *out_1008748947527782496);
void live_H_14(double *state, double *unused, double *out_5792117853049059727);
void live_h_33(double *state, double *unused, double *out_8204202647301706517);
void live_H_33(double *state, double *unused, double *out_5022117979796077237);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}