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
void live_H(double *in_vec, double *out_6365583487552157171);
void live_err_fun(double *nom_x, double *delta_x, double *out_5346167204367436890);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_2579927589579077157);
void live_H_mod_fun(double *state, double *out_5549338810147349673);
void live_f_fun(double *state, double dt, double *out_5240637061299273937);
void live_F_fun(double *state, double dt, double *out_4089696095102840351);
void live_h_4(double *state, double *unused, double *out_39464621260420671);
void live_H_4(double *state, double *unused, double *out_5765764778194156246);
void live_h_9(double *state, double *unused, double *out_1350743087601519525);
void live_H_9(double *state, double *unused, double *out_1521454157070291224);
void live_h_10(double *state, double *unused, double *out_3423593416762292844);
void live_H_10(double *state, double *unused, double *out_5964672752917544224);
void live_h_12(double *state, double *unused, double *out_775274685468455003);
void live_H_12(double *state, double *unused, double *out_6299720918472662374);
void live_h_35(double *state, double *unused, double *out_5295382600475008259);
void live_H_35(double *state, double *unused, double *out_4646926567813307955);
void live_h_32(double *state, double *unused, double *out_7530322781783105718);
void live_H_32(double *state, double *unused, double *out_7998513133688631103);
void live_h_13(double *state, double *unused, double *out_4940020963500413901);
void live_H_13(double *state, double *unused, double *out_6016490829242061086);
void live_h_14(double *state, double *unused, double *out_1350743087601519525);
void live_H_14(double *state, double *unused, double *out_1521454157070291224);
void live_h_33(double *state, double *unused, double *out_6367766257064146397);
void live_H_33(double *state, double *unused, double *out_7797483572452165559);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}