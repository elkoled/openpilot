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
void live_H(double *in_vec, double *out_69714427579637706);
void live_err_fun(double *nom_x, double *delta_x, double *out_5444905409153653681);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_4608305250173598539);
void live_H_mod_fun(double *state, double *out_8972771830906541404);
void live_f_fun(double *state, double dt, double *out_1842217584614110206);
void live_F_fun(double *state, double dt, double *out_5843888478312245686);
void live_h_4(double *state, double *unused, double *out_2830441860469917789);
void live_H_4(double *state, double *unused, double *out_8285336287928576429);
void live_h_9(double *state, double *unused, double *out_7006787553566514297);
void live_H_9(double *state, double *unused, double *out_8526525934558167074);
void live_h_10(double *state, double *unused, double *out_8854198090198982480);
void live_H_10(double *state, double *unused, double *out_5359505665264235307);
void live_h_12(double *state, double *unused, double *out_232787342529916571);
void live_H_12(double *state, double *unused, double *out_5141951377749013392);
void live_h_35(double *state, double *unused, double *out_1877832896197337039);
void live_H_35(double *state, double *unused, double *out_6794745728408367811);
void live_h_32(double *state, double *unused, double *out_1331197872784152550);
void live_H_32(double *state, double *unused, double *out_993441356200755253);
void live_h_13(double *state, double *unused, double *out_985751227432456583);
void live_H_13(double *state, double *unused, double *out_330079723043753288);
void live_h_14(double *state, double *unused, double *out_7006787553566514297);
void live_H_14(double *state, double *unused, double *out_8526525934558167074);
void live_h_33(double *state, double *unused, double *out_8174259301173547701);
void live_H_33(double *state, double *unused, double *out_3644188723769510207);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}