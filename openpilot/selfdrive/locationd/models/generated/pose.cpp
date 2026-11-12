#include "pose.h"

namespace {
#define DIM 18
#define EDIM 18
#define MEDIM 18
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_4 = 7.814727903251177;
const static double MAHA_THRESH_10 = 7.814727903251177;
const static double MAHA_THRESH_13 = 7.814727903251177;
const static double MAHA_THRESH_14 = 7.814727903251177;

/******************************************************************************
 *                      Code generated with SymPy 1.14.0                      *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_7924553503606272438) {
   out_7924553503606272438[0] = delta_x[0] + nom_x[0];
   out_7924553503606272438[1] = delta_x[1] + nom_x[1];
   out_7924553503606272438[2] = delta_x[2] + nom_x[2];
   out_7924553503606272438[3] = delta_x[3] + nom_x[3];
   out_7924553503606272438[4] = delta_x[4] + nom_x[4];
   out_7924553503606272438[5] = delta_x[5] + nom_x[5];
   out_7924553503606272438[6] = delta_x[6] + nom_x[6];
   out_7924553503606272438[7] = delta_x[7] + nom_x[7];
   out_7924553503606272438[8] = delta_x[8] + nom_x[8];
   out_7924553503606272438[9] = delta_x[9] + nom_x[9];
   out_7924553503606272438[10] = delta_x[10] + nom_x[10];
   out_7924553503606272438[11] = delta_x[11] + nom_x[11];
   out_7924553503606272438[12] = delta_x[12] + nom_x[12];
   out_7924553503606272438[13] = delta_x[13] + nom_x[13];
   out_7924553503606272438[14] = delta_x[14] + nom_x[14];
   out_7924553503606272438[15] = delta_x[15] + nom_x[15];
   out_7924553503606272438[16] = delta_x[16] + nom_x[16];
   out_7924553503606272438[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1094996591856183194) {
   out_1094996591856183194[0] = -nom_x[0] + true_x[0];
   out_1094996591856183194[1] = -nom_x[1] + true_x[1];
   out_1094996591856183194[2] = -nom_x[2] + true_x[2];
   out_1094996591856183194[3] = -nom_x[3] + true_x[3];
   out_1094996591856183194[4] = -nom_x[4] + true_x[4];
   out_1094996591856183194[5] = -nom_x[5] + true_x[5];
   out_1094996591856183194[6] = -nom_x[6] + true_x[6];
   out_1094996591856183194[7] = -nom_x[7] + true_x[7];
   out_1094996591856183194[8] = -nom_x[8] + true_x[8];
   out_1094996591856183194[9] = -nom_x[9] + true_x[9];
   out_1094996591856183194[10] = -nom_x[10] + true_x[10];
   out_1094996591856183194[11] = -nom_x[11] + true_x[11];
   out_1094996591856183194[12] = -nom_x[12] + true_x[12];
   out_1094996591856183194[13] = -nom_x[13] + true_x[13];
   out_1094996591856183194[14] = -nom_x[14] + true_x[14];
   out_1094996591856183194[15] = -nom_x[15] + true_x[15];
   out_1094996591856183194[16] = -nom_x[16] + true_x[16];
   out_1094996591856183194[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_5263462732536222298) {
   out_5263462732536222298[0] = 1.0;
   out_5263462732536222298[1] = 0.0;
   out_5263462732536222298[2] = 0.0;
   out_5263462732536222298[3] = 0.0;
   out_5263462732536222298[4] = 0.0;
   out_5263462732536222298[5] = 0.0;
   out_5263462732536222298[6] = 0.0;
   out_5263462732536222298[7] = 0.0;
   out_5263462732536222298[8] = 0.0;
   out_5263462732536222298[9] = 0.0;
   out_5263462732536222298[10] = 0.0;
   out_5263462732536222298[11] = 0.0;
   out_5263462732536222298[12] = 0.0;
   out_5263462732536222298[13] = 0.0;
   out_5263462732536222298[14] = 0.0;
   out_5263462732536222298[15] = 0.0;
   out_5263462732536222298[16] = 0.0;
   out_5263462732536222298[17] = 0.0;
   out_5263462732536222298[18] = 0.0;
   out_5263462732536222298[19] = 1.0;
   out_5263462732536222298[20] = 0.0;
   out_5263462732536222298[21] = 0.0;
   out_5263462732536222298[22] = 0.0;
   out_5263462732536222298[23] = 0.0;
   out_5263462732536222298[24] = 0.0;
   out_5263462732536222298[25] = 0.0;
   out_5263462732536222298[26] = 0.0;
   out_5263462732536222298[27] = 0.0;
   out_5263462732536222298[28] = 0.0;
   out_5263462732536222298[29] = 0.0;
   out_5263462732536222298[30] = 0.0;
   out_5263462732536222298[31] = 0.0;
   out_5263462732536222298[32] = 0.0;
   out_5263462732536222298[33] = 0.0;
   out_5263462732536222298[34] = 0.0;
   out_5263462732536222298[35] = 0.0;
   out_5263462732536222298[36] = 0.0;
   out_5263462732536222298[37] = 0.0;
   out_5263462732536222298[38] = 1.0;
   out_5263462732536222298[39] = 0.0;
   out_5263462732536222298[40] = 0.0;
   out_5263462732536222298[41] = 0.0;
   out_5263462732536222298[42] = 0.0;
   out_5263462732536222298[43] = 0.0;
   out_5263462732536222298[44] = 0.0;
   out_5263462732536222298[45] = 0.0;
   out_5263462732536222298[46] = 0.0;
   out_5263462732536222298[47] = 0.0;
   out_5263462732536222298[48] = 0.0;
   out_5263462732536222298[49] = 0.0;
   out_5263462732536222298[50] = 0.0;
   out_5263462732536222298[51] = 0.0;
   out_5263462732536222298[52] = 0.0;
   out_5263462732536222298[53] = 0.0;
   out_5263462732536222298[54] = 0.0;
   out_5263462732536222298[55] = 0.0;
   out_5263462732536222298[56] = 0.0;
   out_5263462732536222298[57] = 1.0;
   out_5263462732536222298[58] = 0.0;
   out_5263462732536222298[59] = 0.0;
   out_5263462732536222298[60] = 0.0;
   out_5263462732536222298[61] = 0.0;
   out_5263462732536222298[62] = 0.0;
   out_5263462732536222298[63] = 0.0;
   out_5263462732536222298[64] = 0.0;
   out_5263462732536222298[65] = 0.0;
   out_5263462732536222298[66] = 0.0;
   out_5263462732536222298[67] = 0.0;
   out_5263462732536222298[68] = 0.0;
   out_5263462732536222298[69] = 0.0;
   out_5263462732536222298[70] = 0.0;
   out_5263462732536222298[71] = 0.0;
   out_5263462732536222298[72] = 0.0;
   out_5263462732536222298[73] = 0.0;
   out_5263462732536222298[74] = 0.0;
   out_5263462732536222298[75] = 0.0;
   out_5263462732536222298[76] = 1.0;
   out_5263462732536222298[77] = 0.0;
   out_5263462732536222298[78] = 0.0;
   out_5263462732536222298[79] = 0.0;
   out_5263462732536222298[80] = 0.0;
   out_5263462732536222298[81] = 0.0;
   out_5263462732536222298[82] = 0.0;
   out_5263462732536222298[83] = 0.0;
   out_5263462732536222298[84] = 0.0;
   out_5263462732536222298[85] = 0.0;
   out_5263462732536222298[86] = 0.0;
   out_5263462732536222298[87] = 0.0;
   out_5263462732536222298[88] = 0.0;
   out_5263462732536222298[89] = 0.0;
   out_5263462732536222298[90] = 0.0;
   out_5263462732536222298[91] = 0.0;
   out_5263462732536222298[92] = 0.0;
   out_5263462732536222298[93] = 0.0;
   out_5263462732536222298[94] = 0.0;
   out_5263462732536222298[95] = 1.0;
   out_5263462732536222298[96] = 0.0;
   out_5263462732536222298[97] = 0.0;
   out_5263462732536222298[98] = 0.0;
   out_5263462732536222298[99] = 0.0;
   out_5263462732536222298[100] = 0.0;
   out_5263462732536222298[101] = 0.0;
   out_5263462732536222298[102] = 0.0;
   out_5263462732536222298[103] = 0.0;
   out_5263462732536222298[104] = 0.0;
   out_5263462732536222298[105] = 0.0;
   out_5263462732536222298[106] = 0.0;
   out_5263462732536222298[107] = 0.0;
   out_5263462732536222298[108] = 0.0;
   out_5263462732536222298[109] = 0.0;
   out_5263462732536222298[110] = 0.0;
   out_5263462732536222298[111] = 0.0;
   out_5263462732536222298[112] = 0.0;
   out_5263462732536222298[113] = 0.0;
   out_5263462732536222298[114] = 1.0;
   out_5263462732536222298[115] = 0.0;
   out_5263462732536222298[116] = 0.0;
   out_5263462732536222298[117] = 0.0;
   out_5263462732536222298[118] = 0.0;
   out_5263462732536222298[119] = 0.0;
   out_5263462732536222298[120] = 0.0;
   out_5263462732536222298[121] = 0.0;
   out_5263462732536222298[122] = 0.0;
   out_5263462732536222298[123] = 0.0;
   out_5263462732536222298[124] = 0.0;
   out_5263462732536222298[125] = 0.0;
   out_5263462732536222298[126] = 0.0;
   out_5263462732536222298[127] = 0.0;
   out_5263462732536222298[128] = 0.0;
   out_5263462732536222298[129] = 0.0;
   out_5263462732536222298[130] = 0.0;
   out_5263462732536222298[131] = 0.0;
   out_5263462732536222298[132] = 0.0;
   out_5263462732536222298[133] = 1.0;
   out_5263462732536222298[134] = 0.0;
   out_5263462732536222298[135] = 0.0;
   out_5263462732536222298[136] = 0.0;
   out_5263462732536222298[137] = 0.0;
   out_5263462732536222298[138] = 0.0;
   out_5263462732536222298[139] = 0.0;
   out_5263462732536222298[140] = 0.0;
   out_5263462732536222298[141] = 0.0;
   out_5263462732536222298[142] = 0.0;
   out_5263462732536222298[143] = 0.0;
   out_5263462732536222298[144] = 0.0;
   out_5263462732536222298[145] = 0.0;
   out_5263462732536222298[146] = 0.0;
   out_5263462732536222298[147] = 0.0;
   out_5263462732536222298[148] = 0.0;
   out_5263462732536222298[149] = 0.0;
   out_5263462732536222298[150] = 0.0;
   out_5263462732536222298[151] = 0.0;
   out_5263462732536222298[152] = 1.0;
   out_5263462732536222298[153] = 0.0;
   out_5263462732536222298[154] = 0.0;
   out_5263462732536222298[155] = 0.0;
   out_5263462732536222298[156] = 0.0;
   out_5263462732536222298[157] = 0.0;
   out_5263462732536222298[158] = 0.0;
   out_5263462732536222298[159] = 0.0;
   out_5263462732536222298[160] = 0.0;
   out_5263462732536222298[161] = 0.0;
   out_5263462732536222298[162] = 0.0;
   out_5263462732536222298[163] = 0.0;
   out_5263462732536222298[164] = 0.0;
   out_5263462732536222298[165] = 0.0;
   out_5263462732536222298[166] = 0.0;
   out_5263462732536222298[167] = 0.0;
   out_5263462732536222298[168] = 0.0;
   out_5263462732536222298[169] = 0.0;
   out_5263462732536222298[170] = 0.0;
   out_5263462732536222298[171] = 1.0;
   out_5263462732536222298[172] = 0.0;
   out_5263462732536222298[173] = 0.0;
   out_5263462732536222298[174] = 0.0;
   out_5263462732536222298[175] = 0.0;
   out_5263462732536222298[176] = 0.0;
   out_5263462732536222298[177] = 0.0;
   out_5263462732536222298[178] = 0.0;
   out_5263462732536222298[179] = 0.0;
   out_5263462732536222298[180] = 0.0;
   out_5263462732536222298[181] = 0.0;
   out_5263462732536222298[182] = 0.0;
   out_5263462732536222298[183] = 0.0;
   out_5263462732536222298[184] = 0.0;
   out_5263462732536222298[185] = 0.0;
   out_5263462732536222298[186] = 0.0;
   out_5263462732536222298[187] = 0.0;
   out_5263462732536222298[188] = 0.0;
   out_5263462732536222298[189] = 0.0;
   out_5263462732536222298[190] = 1.0;
   out_5263462732536222298[191] = 0.0;
   out_5263462732536222298[192] = 0.0;
   out_5263462732536222298[193] = 0.0;
   out_5263462732536222298[194] = 0.0;
   out_5263462732536222298[195] = 0.0;
   out_5263462732536222298[196] = 0.0;
   out_5263462732536222298[197] = 0.0;
   out_5263462732536222298[198] = 0.0;
   out_5263462732536222298[199] = 0.0;
   out_5263462732536222298[200] = 0.0;
   out_5263462732536222298[201] = 0.0;
   out_5263462732536222298[202] = 0.0;
   out_5263462732536222298[203] = 0.0;
   out_5263462732536222298[204] = 0.0;
   out_5263462732536222298[205] = 0.0;
   out_5263462732536222298[206] = 0.0;
   out_5263462732536222298[207] = 0.0;
   out_5263462732536222298[208] = 0.0;
   out_5263462732536222298[209] = 1.0;
   out_5263462732536222298[210] = 0.0;
   out_5263462732536222298[211] = 0.0;
   out_5263462732536222298[212] = 0.0;
   out_5263462732536222298[213] = 0.0;
   out_5263462732536222298[214] = 0.0;
   out_5263462732536222298[215] = 0.0;
   out_5263462732536222298[216] = 0.0;
   out_5263462732536222298[217] = 0.0;
   out_5263462732536222298[218] = 0.0;
   out_5263462732536222298[219] = 0.0;
   out_5263462732536222298[220] = 0.0;
   out_5263462732536222298[221] = 0.0;
   out_5263462732536222298[222] = 0.0;
   out_5263462732536222298[223] = 0.0;
   out_5263462732536222298[224] = 0.0;
   out_5263462732536222298[225] = 0.0;
   out_5263462732536222298[226] = 0.0;
   out_5263462732536222298[227] = 0.0;
   out_5263462732536222298[228] = 1.0;
   out_5263462732536222298[229] = 0.0;
   out_5263462732536222298[230] = 0.0;
   out_5263462732536222298[231] = 0.0;
   out_5263462732536222298[232] = 0.0;
   out_5263462732536222298[233] = 0.0;
   out_5263462732536222298[234] = 0.0;
   out_5263462732536222298[235] = 0.0;
   out_5263462732536222298[236] = 0.0;
   out_5263462732536222298[237] = 0.0;
   out_5263462732536222298[238] = 0.0;
   out_5263462732536222298[239] = 0.0;
   out_5263462732536222298[240] = 0.0;
   out_5263462732536222298[241] = 0.0;
   out_5263462732536222298[242] = 0.0;
   out_5263462732536222298[243] = 0.0;
   out_5263462732536222298[244] = 0.0;
   out_5263462732536222298[245] = 0.0;
   out_5263462732536222298[246] = 0.0;
   out_5263462732536222298[247] = 1.0;
   out_5263462732536222298[248] = 0.0;
   out_5263462732536222298[249] = 0.0;
   out_5263462732536222298[250] = 0.0;
   out_5263462732536222298[251] = 0.0;
   out_5263462732536222298[252] = 0.0;
   out_5263462732536222298[253] = 0.0;
   out_5263462732536222298[254] = 0.0;
   out_5263462732536222298[255] = 0.0;
   out_5263462732536222298[256] = 0.0;
   out_5263462732536222298[257] = 0.0;
   out_5263462732536222298[258] = 0.0;
   out_5263462732536222298[259] = 0.0;
   out_5263462732536222298[260] = 0.0;
   out_5263462732536222298[261] = 0.0;
   out_5263462732536222298[262] = 0.0;
   out_5263462732536222298[263] = 0.0;
   out_5263462732536222298[264] = 0.0;
   out_5263462732536222298[265] = 0.0;
   out_5263462732536222298[266] = 1.0;
   out_5263462732536222298[267] = 0.0;
   out_5263462732536222298[268] = 0.0;
   out_5263462732536222298[269] = 0.0;
   out_5263462732536222298[270] = 0.0;
   out_5263462732536222298[271] = 0.0;
   out_5263462732536222298[272] = 0.0;
   out_5263462732536222298[273] = 0.0;
   out_5263462732536222298[274] = 0.0;
   out_5263462732536222298[275] = 0.0;
   out_5263462732536222298[276] = 0.0;
   out_5263462732536222298[277] = 0.0;
   out_5263462732536222298[278] = 0.0;
   out_5263462732536222298[279] = 0.0;
   out_5263462732536222298[280] = 0.0;
   out_5263462732536222298[281] = 0.0;
   out_5263462732536222298[282] = 0.0;
   out_5263462732536222298[283] = 0.0;
   out_5263462732536222298[284] = 0.0;
   out_5263462732536222298[285] = 1.0;
   out_5263462732536222298[286] = 0.0;
   out_5263462732536222298[287] = 0.0;
   out_5263462732536222298[288] = 0.0;
   out_5263462732536222298[289] = 0.0;
   out_5263462732536222298[290] = 0.0;
   out_5263462732536222298[291] = 0.0;
   out_5263462732536222298[292] = 0.0;
   out_5263462732536222298[293] = 0.0;
   out_5263462732536222298[294] = 0.0;
   out_5263462732536222298[295] = 0.0;
   out_5263462732536222298[296] = 0.0;
   out_5263462732536222298[297] = 0.0;
   out_5263462732536222298[298] = 0.0;
   out_5263462732536222298[299] = 0.0;
   out_5263462732536222298[300] = 0.0;
   out_5263462732536222298[301] = 0.0;
   out_5263462732536222298[302] = 0.0;
   out_5263462732536222298[303] = 0.0;
   out_5263462732536222298[304] = 1.0;
   out_5263462732536222298[305] = 0.0;
   out_5263462732536222298[306] = 0.0;
   out_5263462732536222298[307] = 0.0;
   out_5263462732536222298[308] = 0.0;
   out_5263462732536222298[309] = 0.0;
   out_5263462732536222298[310] = 0.0;
   out_5263462732536222298[311] = 0.0;
   out_5263462732536222298[312] = 0.0;
   out_5263462732536222298[313] = 0.0;
   out_5263462732536222298[314] = 0.0;
   out_5263462732536222298[315] = 0.0;
   out_5263462732536222298[316] = 0.0;
   out_5263462732536222298[317] = 0.0;
   out_5263462732536222298[318] = 0.0;
   out_5263462732536222298[319] = 0.0;
   out_5263462732536222298[320] = 0.0;
   out_5263462732536222298[321] = 0.0;
   out_5263462732536222298[322] = 0.0;
   out_5263462732536222298[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5332214133538366933) {
   out_5332214133538366933[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5332214133538366933[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5332214133538366933[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5332214133538366933[3] = dt*state[12] + state[3];
   out_5332214133538366933[4] = dt*state[13] + state[4];
   out_5332214133538366933[5] = dt*state[14] + state[5];
   out_5332214133538366933[6] = state[6];
   out_5332214133538366933[7] = state[7];
   out_5332214133538366933[8] = state[8];
   out_5332214133538366933[9] = state[9];
   out_5332214133538366933[10] = state[10];
   out_5332214133538366933[11] = state[11];
   out_5332214133538366933[12] = state[12];
   out_5332214133538366933[13] = state[13];
   out_5332214133538366933[14] = state[14];
   out_5332214133538366933[15] = state[15];
   out_5332214133538366933[16] = state[16];
   out_5332214133538366933[17] = state[17];
}
void F_fun(double *state, double dt, double *out_4992300284098693487) {
   out_4992300284098693487[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4992300284098693487[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4992300284098693487[2] = 0;
   out_4992300284098693487[3] = 0;
   out_4992300284098693487[4] = 0;
   out_4992300284098693487[5] = 0;
   out_4992300284098693487[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4992300284098693487[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4992300284098693487[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4992300284098693487[9] = 0;
   out_4992300284098693487[10] = 0;
   out_4992300284098693487[11] = 0;
   out_4992300284098693487[12] = 0;
   out_4992300284098693487[13] = 0;
   out_4992300284098693487[14] = 0;
   out_4992300284098693487[15] = 0;
   out_4992300284098693487[16] = 0;
   out_4992300284098693487[17] = 0;
   out_4992300284098693487[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4992300284098693487[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4992300284098693487[20] = 0;
   out_4992300284098693487[21] = 0;
   out_4992300284098693487[22] = 0;
   out_4992300284098693487[23] = 0;
   out_4992300284098693487[24] = 0;
   out_4992300284098693487[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4992300284098693487[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4992300284098693487[27] = 0;
   out_4992300284098693487[28] = 0;
   out_4992300284098693487[29] = 0;
   out_4992300284098693487[30] = 0;
   out_4992300284098693487[31] = 0;
   out_4992300284098693487[32] = 0;
   out_4992300284098693487[33] = 0;
   out_4992300284098693487[34] = 0;
   out_4992300284098693487[35] = 0;
   out_4992300284098693487[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4992300284098693487[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4992300284098693487[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4992300284098693487[39] = 0;
   out_4992300284098693487[40] = 0;
   out_4992300284098693487[41] = 0;
   out_4992300284098693487[42] = 0;
   out_4992300284098693487[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4992300284098693487[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4992300284098693487[45] = 0;
   out_4992300284098693487[46] = 0;
   out_4992300284098693487[47] = 0;
   out_4992300284098693487[48] = 0;
   out_4992300284098693487[49] = 0;
   out_4992300284098693487[50] = 0;
   out_4992300284098693487[51] = 0;
   out_4992300284098693487[52] = 0;
   out_4992300284098693487[53] = 0;
   out_4992300284098693487[54] = 0;
   out_4992300284098693487[55] = 0;
   out_4992300284098693487[56] = 0;
   out_4992300284098693487[57] = 1;
   out_4992300284098693487[58] = 0;
   out_4992300284098693487[59] = 0;
   out_4992300284098693487[60] = 0;
   out_4992300284098693487[61] = 0;
   out_4992300284098693487[62] = 0;
   out_4992300284098693487[63] = 0;
   out_4992300284098693487[64] = 0;
   out_4992300284098693487[65] = 0;
   out_4992300284098693487[66] = dt;
   out_4992300284098693487[67] = 0;
   out_4992300284098693487[68] = 0;
   out_4992300284098693487[69] = 0;
   out_4992300284098693487[70] = 0;
   out_4992300284098693487[71] = 0;
   out_4992300284098693487[72] = 0;
   out_4992300284098693487[73] = 0;
   out_4992300284098693487[74] = 0;
   out_4992300284098693487[75] = 0;
   out_4992300284098693487[76] = 1;
   out_4992300284098693487[77] = 0;
   out_4992300284098693487[78] = 0;
   out_4992300284098693487[79] = 0;
   out_4992300284098693487[80] = 0;
   out_4992300284098693487[81] = 0;
   out_4992300284098693487[82] = 0;
   out_4992300284098693487[83] = 0;
   out_4992300284098693487[84] = 0;
   out_4992300284098693487[85] = dt;
   out_4992300284098693487[86] = 0;
   out_4992300284098693487[87] = 0;
   out_4992300284098693487[88] = 0;
   out_4992300284098693487[89] = 0;
   out_4992300284098693487[90] = 0;
   out_4992300284098693487[91] = 0;
   out_4992300284098693487[92] = 0;
   out_4992300284098693487[93] = 0;
   out_4992300284098693487[94] = 0;
   out_4992300284098693487[95] = 1;
   out_4992300284098693487[96] = 0;
   out_4992300284098693487[97] = 0;
   out_4992300284098693487[98] = 0;
   out_4992300284098693487[99] = 0;
   out_4992300284098693487[100] = 0;
   out_4992300284098693487[101] = 0;
   out_4992300284098693487[102] = 0;
   out_4992300284098693487[103] = 0;
   out_4992300284098693487[104] = dt;
   out_4992300284098693487[105] = 0;
   out_4992300284098693487[106] = 0;
   out_4992300284098693487[107] = 0;
   out_4992300284098693487[108] = 0;
   out_4992300284098693487[109] = 0;
   out_4992300284098693487[110] = 0;
   out_4992300284098693487[111] = 0;
   out_4992300284098693487[112] = 0;
   out_4992300284098693487[113] = 0;
   out_4992300284098693487[114] = 1;
   out_4992300284098693487[115] = 0;
   out_4992300284098693487[116] = 0;
   out_4992300284098693487[117] = 0;
   out_4992300284098693487[118] = 0;
   out_4992300284098693487[119] = 0;
   out_4992300284098693487[120] = 0;
   out_4992300284098693487[121] = 0;
   out_4992300284098693487[122] = 0;
   out_4992300284098693487[123] = 0;
   out_4992300284098693487[124] = 0;
   out_4992300284098693487[125] = 0;
   out_4992300284098693487[126] = 0;
   out_4992300284098693487[127] = 0;
   out_4992300284098693487[128] = 0;
   out_4992300284098693487[129] = 0;
   out_4992300284098693487[130] = 0;
   out_4992300284098693487[131] = 0;
   out_4992300284098693487[132] = 0;
   out_4992300284098693487[133] = 1;
   out_4992300284098693487[134] = 0;
   out_4992300284098693487[135] = 0;
   out_4992300284098693487[136] = 0;
   out_4992300284098693487[137] = 0;
   out_4992300284098693487[138] = 0;
   out_4992300284098693487[139] = 0;
   out_4992300284098693487[140] = 0;
   out_4992300284098693487[141] = 0;
   out_4992300284098693487[142] = 0;
   out_4992300284098693487[143] = 0;
   out_4992300284098693487[144] = 0;
   out_4992300284098693487[145] = 0;
   out_4992300284098693487[146] = 0;
   out_4992300284098693487[147] = 0;
   out_4992300284098693487[148] = 0;
   out_4992300284098693487[149] = 0;
   out_4992300284098693487[150] = 0;
   out_4992300284098693487[151] = 0;
   out_4992300284098693487[152] = 1;
   out_4992300284098693487[153] = 0;
   out_4992300284098693487[154] = 0;
   out_4992300284098693487[155] = 0;
   out_4992300284098693487[156] = 0;
   out_4992300284098693487[157] = 0;
   out_4992300284098693487[158] = 0;
   out_4992300284098693487[159] = 0;
   out_4992300284098693487[160] = 0;
   out_4992300284098693487[161] = 0;
   out_4992300284098693487[162] = 0;
   out_4992300284098693487[163] = 0;
   out_4992300284098693487[164] = 0;
   out_4992300284098693487[165] = 0;
   out_4992300284098693487[166] = 0;
   out_4992300284098693487[167] = 0;
   out_4992300284098693487[168] = 0;
   out_4992300284098693487[169] = 0;
   out_4992300284098693487[170] = 0;
   out_4992300284098693487[171] = 1;
   out_4992300284098693487[172] = 0;
   out_4992300284098693487[173] = 0;
   out_4992300284098693487[174] = 0;
   out_4992300284098693487[175] = 0;
   out_4992300284098693487[176] = 0;
   out_4992300284098693487[177] = 0;
   out_4992300284098693487[178] = 0;
   out_4992300284098693487[179] = 0;
   out_4992300284098693487[180] = 0;
   out_4992300284098693487[181] = 0;
   out_4992300284098693487[182] = 0;
   out_4992300284098693487[183] = 0;
   out_4992300284098693487[184] = 0;
   out_4992300284098693487[185] = 0;
   out_4992300284098693487[186] = 0;
   out_4992300284098693487[187] = 0;
   out_4992300284098693487[188] = 0;
   out_4992300284098693487[189] = 0;
   out_4992300284098693487[190] = 1;
   out_4992300284098693487[191] = 0;
   out_4992300284098693487[192] = 0;
   out_4992300284098693487[193] = 0;
   out_4992300284098693487[194] = 0;
   out_4992300284098693487[195] = 0;
   out_4992300284098693487[196] = 0;
   out_4992300284098693487[197] = 0;
   out_4992300284098693487[198] = 0;
   out_4992300284098693487[199] = 0;
   out_4992300284098693487[200] = 0;
   out_4992300284098693487[201] = 0;
   out_4992300284098693487[202] = 0;
   out_4992300284098693487[203] = 0;
   out_4992300284098693487[204] = 0;
   out_4992300284098693487[205] = 0;
   out_4992300284098693487[206] = 0;
   out_4992300284098693487[207] = 0;
   out_4992300284098693487[208] = 0;
   out_4992300284098693487[209] = 1;
   out_4992300284098693487[210] = 0;
   out_4992300284098693487[211] = 0;
   out_4992300284098693487[212] = 0;
   out_4992300284098693487[213] = 0;
   out_4992300284098693487[214] = 0;
   out_4992300284098693487[215] = 0;
   out_4992300284098693487[216] = 0;
   out_4992300284098693487[217] = 0;
   out_4992300284098693487[218] = 0;
   out_4992300284098693487[219] = 0;
   out_4992300284098693487[220] = 0;
   out_4992300284098693487[221] = 0;
   out_4992300284098693487[222] = 0;
   out_4992300284098693487[223] = 0;
   out_4992300284098693487[224] = 0;
   out_4992300284098693487[225] = 0;
   out_4992300284098693487[226] = 0;
   out_4992300284098693487[227] = 0;
   out_4992300284098693487[228] = 1;
   out_4992300284098693487[229] = 0;
   out_4992300284098693487[230] = 0;
   out_4992300284098693487[231] = 0;
   out_4992300284098693487[232] = 0;
   out_4992300284098693487[233] = 0;
   out_4992300284098693487[234] = 0;
   out_4992300284098693487[235] = 0;
   out_4992300284098693487[236] = 0;
   out_4992300284098693487[237] = 0;
   out_4992300284098693487[238] = 0;
   out_4992300284098693487[239] = 0;
   out_4992300284098693487[240] = 0;
   out_4992300284098693487[241] = 0;
   out_4992300284098693487[242] = 0;
   out_4992300284098693487[243] = 0;
   out_4992300284098693487[244] = 0;
   out_4992300284098693487[245] = 0;
   out_4992300284098693487[246] = 0;
   out_4992300284098693487[247] = 1;
   out_4992300284098693487[248] = 0;
   out_4992300284098693487[249] = 0;
   out_4992300284098693487[250] = 0;
   out_4992300284098693487[251] = 0;
   out_4992300284098693487[252] = 0;
   out_4992300284098693487[253] = 0;
   out_4992300284098693487[254] = 0;
   out_4992300284098693487[255] = 0;
   out_4992300284098693487[256] = 0;
   out_4992300284098693487[257] = 0;
   out_4992300284098693487[258] = 0;
   out_4992300284098693487[259] = 0;
   out_4992300284098693487[260] = 0;
   out_4992300284098693487[261] = 0;
   out_4992300284098693487[262] = 0;
   out_4992300284098693487[263] = 0;
   out_4992300284098693487[264] = 0;
   out_4992300284098693487[265] = 0;
   out_4992300284098693487[266] = 1;
   out_4992300284098693487[267] = 0;
   out_4992300284098693487[268] = 0;
   out_4992300284098693487[269] = 0;
   out_4992300284098693487[270] = 0;
   out_4992300284098693487[271] = 0;
   out_4992300284098693487[272] = 0;
   out_4992300284098693487[273] = 0;
   out_4992300284098693487[274] = 0;
   out_4992300284098693487[275] = 0;
   out_4992300284098693487[276] = 0;
   out_4992300284098693487[277] = 0;
   out_4992300284098693487[278] = 0;
   out_4992300284098693487[279] = 0;
   out_4992300284098693487[280] = 0;
   out_4992300284098693487[281] = 0;
   out_4992300284098693487[282] = 0;
   out_4992300284098693487[283] = 0;
   out_4992300284098693487[284] = 0;
   out_4992300284098693487[285] = 1;
   out_4992300284098693487[286] = 0;
   out_4992300284098693487[287] = 0;
   out_4992300284098693487[288] = 0;
   out_4992300284098693487[289] = 0;
   out_4992300284098693487[290] = 0;
   out_4992300284098693487[291] = 0;
   out_4992300284098693487[292] = 0;
   out_4992300284098693487[293] = 0;
   out_4992300284098693487[294] = 0;
   out_4992300284098693487[295] = 0;
   out_4992300284098693487[296] = 0;
   out_4992300284098693487[297] = 0;
   out_4992300284098693487[298] = 0;
   out_4992300284098693487[299] = 0;
   out_4992300284098693487[300] = 0;
   out_4992300284098693487[301] = 0;
   out_4992300284098693487[302] = 0;
   out_4992300284098693487[303] = 0;
   out_4992300284098693487[304] = 1;
   out_4992300284098693487[305] = 0;
   out_4992300284098693487[306] = 0;
   out_4992300284098693487[307] = 0;
   out_4992300284098693487[308] = 0;
   out_4992300284098693487[309] = 0;
   out_4992300284098693487[310] = 0;
   out_4992300284098693487[311] = 0;
   out_4992300284098693487[312] = 0;
   out_4992300284098693487[313] = 0;
   out_4992300284098693487[314] = 0;
   out_4992300284098693487[315] = 0;
   out_4992300284098693487[316] = 0;
   out_4992300284098693487[317] = 0;
   out_4992300284098693487[318] = 0;
   out_4992300284098693487[319] = 0;
   out_4992300284098693487[320] = 0;
   out_4992300284098693487[321] = 0;
   out_4992300284098693487[322] = 0;
   out_4992300284098693487[323] = 1;
}
void h_4(double *state, double *unused, double *out_4840607086056089263) {
   out_4840607086056089263[0] = state[6] + state[9];
   out_4840607086056089263[1] = state[7] + state[10];
   out_4840607086056089263[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7083161458442519159) {
   out_7083161458442519159[0] = 0;
   out_7083161458442519159[1] = 0;
   out_7083161458442519159[2] = 0;
   out_7083161458442519159[3] = 0;
   out_7083161458442519159[4] = 0;
   out_7083161458442519159[5] = 0;
   out_7083161458442519159[6] = 1;
   out_7083161458442519159[7] = 0;
   out_7083161458442519159[8] = 0;
   out_7083161458442519159[9] = 1;
   out_7083161458442519159[10] = 0;
   out_7083161458442519159[11] = 0;
   out_7083161458442519159[12] = 0;
   out_7083161458442519159[13] = 0;
   out_7083161458442519159[14] = 0;
   out_7083161458442519159[15] = 0;
   out_7083161458442519159[16] = 0;
   out_7083161458442519159[17] = 0;
   out_7083161458442519159[18] = 0;
   out_7083161458442519159[19] = 0;
   out_7083161458442519159[20] = 0;
   out_7083161458442519159[21] = 0;
   out_7083161458442519159[22] = 0;
   out_7083161458442519159[23] = 0;
   out_7083161458442519159[24] = 0;
   out_7083161458442519159[25] = 1;
   out_7083161458442519159[26] = 0;
   out_7083161458442519159[27] = 0;
   out_7083161458442519159[28] = 1;
   out_7083161458442519159[29] = 0;
   out_7083161458442519159[30] = 0;
   out_7083161458442519159[31] = 0;
   out_7083161458442519159[32] = 0;
   out_7083161458442519159[33] = 0;
   out_7083161458442519159[34] = 0;
   out_7083161458442519159[35] = 0;
   out_7083161458442519159[36] = 0;
   out_7083161458442519159[37] = 0;
   out_7083161458442519159[38] = 0;
   out_7083161458442519159[39] = 0;
   out_7083161458442519159[40] = 0;
   out_7083161458442519159[41] = 0;
   out_7083161458442519159[42] = 0;
   out_7083161458442519159[43] = 0;
   out_7083161458442519159[44] = 1;
   out_7083161458442519159[45] = 0;
   out_7083161458442519159[46] = 0;
   out_7083161458442519159[47] = 1;
   out_7083161458442519159[48] = 0;
   out_7083161458442519159[49] = 0;
   out_7083161458442519159[50] = 0;
   out_7083161458442519159[51] = 0;
   out_7083161458442519159[52] = 0;
   out_7083161458442519159[53] = 0;
}
void h_10(double *state, double *unused, double *out_4068132626889395165) {
   out_4068132626889395165[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_4068132626889395165[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_4068132626889395165[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_8672064356962455792) {
   out_8672064356962455792[0] = 0;
   out_8672064356962455792[1] = 9.8100000000000005*cos(state[1]);
   out_8672064356962455792[2] = 0;
   out_8672064356962455792[3] = 0;
   out_8672064356962455792[4] = -state[8];
   out_8672064356962455792[5] = state[7];
   out_8672064356962455792[6] = 0;
   out_8672064356962455792[7] = state[5];
   out_8672064356962455792[8] = -state[4];
   out_8672064356962455792[9] = 0;
   out_8672064356962455792[10] = 0;
   out_8672064356962455792[11] = 0;
   out_8672064356962455792[12] = 1;
   out_8672064356962455792[13] = 0;
   out_8672064356962455792[14] = 0;
   out_8672064356962455792[15] = 1;
   out_8672064356962455792[16] = 0;
   out_8672064356962455792[17] = 0;
   out_8672064356962455792[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_8672064356962455792[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_8672064356962455792[20] = 0;
   out_8672064356962455792[21] = state[8];
   out_8672064356962455792[22] = 0;
   out_8672064356962455792[23] = -state[6];
   out_8672064356962455792[24] = -state[5];
   out_8672064356962455792[25] = 0;
   out_8672064356962455792[26] = state[3];
   out_8672064356962455792[27] = 0;
   out_8672064356962455792[28] = 0;
   out_8672064356962455792[29] = 0;
   out_8672064356962455792[30] = 0;
   out_8672064356962455792[31] = 1;
   out_8672064356962455792[32] = 0;
   out_8672064356962455792[33] = 0;
   out_8672064356962455792[34] = 1;
   out_8672064356962455792[35] = 0;
   out_8672064356962455792[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_8672064356962455792[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_8672064356962455792[38] = 0;
   out_8672064356962455792[39] = -state[7];
   out_8672064356962455792[40] = state[6];
   out_8672064356962455792[41] = 0;
   out_8672064356962455792[42] = state[4];
   out_8672064356962455792[43] = -state[3];
   out_8672064356962455792[44] = 0;
   out_8672064356962455792[45] = 0;
   out_8672064356962455792[46] = 0;
   out_8672064356962455792[47] = 0;
   out_8672064356962455792[48] = 0;
   out_8672064356962455792[49] = 0;
   out_8672064356962455792[50] = 1;
   out_8672064356962455792[51] = 0;
   out_8672064356962455792[52] = 0;
   out_8672064356962455792[53] = 1;
}
void h_13(double *state, double *unused, double *out_7526180503868782014) {
   out_7526180503868782014[0] = state[3];
   out_7526180503868782014[1] = state[4];
   out_7526180503868782014[2] = state[5];
}
void H_13(double *state, double *unused, double *out_527469749874181770) {
   out_527469749874181770[0] = 0;
   out_527469749874181770[1] = 0;
   out_527469749874181770[2] = 0;
   out_527469749874181770[3] = 1;
   out_527469749874181770[4] = 0;
   out_527469749874181770[5] = 0;
   out_527469749874181770[6] = 0;
   out_527469749874181770[7] = 0;
   out_527469749874181770[8] = 0;
   out_527469749874181770[9] = 0;
   out_527469749874181770[10] = 0;
   out_527469749874181770[11] = 0;
   out_527469749874181770[12] = 0;
   out_527469749874181770[13] = 0;
   out_527469749874181770[14] = 0;
   out_527469749874181770[15] = 0;
   out_527469749874181770[16] = 0;
   out_527469749874181770[17] = 0;
   out_527469749874181770[18] = 0;
   out_527469749874181770[19] = 0;
   out_527469749874181770[20] = 0;
   out_527469749874181770[21] = 0;
   out_527469749874181770[22] = 1;
   out_527469749874181770[23] = 0;
   out_527469749874181770[24] = 0;
   out_527469749874181770[25] = 0;
   out_527469749874181770[26] = 0;
   out_527469749874181770[27] = 0;
   out_527469749874181770[28] = 0;
   out_527469749874181770[29] = 0;
   out_527469749874181770[30] = 0;
   out_527469749874181770[31] = 0;
   out_527469749874181770[32] = 0;
   out_527469749874181770[33] = 0;
   out_527469749874181770[34] = 0;
   out_527469749874181770[35] = 0;
   out_527469749874181770[36] = 0;
   out_527469749874181770[37] = 0;
   out_527469749874181770[38] = 0;
   out_527469749874181770[39] = 0;
   out_527469749874181770[40] = 0;
   out_527469749874181770[41] = 1;
   out_527469749874181770[42] = 0;
   out_527469749874181770[43] = 0;
   out_527469749874181770[44] = 0;
   out_527469749874181770[45] = 0;
   out_527469749874181770[46] = 0;
   out_527469749874181770[47] = 0;
   out_527469749874181770[48] = 0;
   out_527469749874181770[49] = 0;
   out_527469749874181770[50] = 0;
   out_527469749874181770[51] = 0;
   out_527469749874181770[52] = 0;
   out_527469749874181770[53] = 0;
}
void h_14(double *state, double *unused, double *out_2452990038371875929) {
   out_2452990038371875929[0] = state[6];
   out_2452990038371875929[1] = state[7];
   out_2452990038371875929[2] = state[8];
}
void H_14(double *state, double *unused, double *out_3119920602103034630) {
   out_3119920602103034630[0] = 0;
   out_3119920602103034630[1] = 0;
   out_3119920602103034630[2] = 0;
   out_3119920602103034630[3] = 0;
   out_3119920602103034630[4] = 0;
   out_3119920602103034630[5] = 0;
   out_3119920602103034630[6] = 1;
   out_3119920602103034630[7] = 0;
   out_3119920602103034630[8] = 0;
   out_3119920602103034630[9] = 0;
   out_3119920602103034630[10] = 0;
   out_3119920602103034630[11] = 0;
   out_3119920602103034630[12] = 0;
   out_3119920602103034630[13] = 0;
   out_3119920602103034630[14] = 0;
   out_3119920602103034630[15] = 0;
   out_3119920602103034630[16] = 0;
   out_3119920602103034630[17] = 0;
   out_3119920602103034630[18] = 0;
   out_3119920602103034630[19] = 0;
   out_3119920602103034630[20] = 0;
   out_3119920602103034630[21] = 0;
   out_3119920602103034630[22] = 0;
   out_3119920602103034630[23] = 0;
   out_3119920602103034630[24] = 0;
   out_3119920602103034630[25] = 1;
   out_3119920602103034630[26] = 0;
   out_3119920602103034630[27] = 0;
   out_3119920602103034630[28] = 0;
   out_3119920602103034630[29] = 0;
   out_3119920602103034630[30] = 0;
   out_3119920602103034630[31] = 0;
   out_3119920602103034630[32] = 0;
   out_3119920602103034630[33] = 0;
   out_3119920602103034630[34] = 0;
   out_3119920602103034630[35] = 0;
   out_3119920602103034630[36] = 0;
   out_3119920602103034630[37] = 0;
   out_3119920602103034630[38] = 0;
   out_3119920602103034630[39] = 0;
   out_3119920602103034630[40] = 0;
   out_3119920602103034630[41] = 0;
   out_3119920602103034630[42] = 0;
   out_3119920602103034630[43] = 0;
   out_3119920602103034630[44] = 1;
   out_3119920602103034630[45] = 0;
   out_3119920602103034630[46] = 0;
   out_3119920602103034630[47] = 0;
   out_3119920602103034630[48] = 0;
   out_3119920602103034630[49] = 0;
   out_3119920602103034630[50] = 0;
   out_3119920602103034630[51] = 0;
   out_3119920602103034630[52] = 0;
   out_3119920602103034630[53] = 0;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void pose_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_4, H_4, NULL, in_z, in_R, in_ea, MAHA_THRESH_4);
}
void pose_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_10, H_10, NULL, in_z, in_R, in_ea, MAHA_THRESH_10);
}
void pose_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_13, H_13, NULL, in_z, in_R, in_ea, MAHA_THRESH_13);
}
void pose_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<3, 3, 0>(in_x, in_P, h_14, H_14, NULL, in_z, in_R, in_ea, MAHA_THRESH_14);
}
void pose_err_fun(double *nom_x, double *delta_x, double *out_7924553503606272438) {
  err_fun(nom_x, delta_x, out_7924553503606272438);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1094996591856183194) {
  inv_err_fun(nom_x, true_x, out_1094996591856183194);
}
void pose_H_mod_fun(double *state, double *out_5263462732536222298) {
  H_mod_fun(state, out_5263462732536222298);
}
void pose_f_fun(double *state, double dt, double *out_5332214133538366933) {
  f_fun(state,  dt, out_5332214133538366933);
}
void pose_F_fun(double *state, double dt, double *out_4992300284098693487) {
  F_fun(state,  dt, out_4992300284098693487);
}
void pose_h_4(double *state, double *unused, double *out_4840607086056089263) {
  h_4(state, unused, out_4840607086056089263);
}
void pose_H_4(double *state, double *unused, double *out_7083161458442519159) {
  H_4(state, unused, out_7083161458442519159);
}
void pose_h_10(double *state, double *unused, double *out_4068132626889395165) {
  h_10(state, unused, out_4068132626889395165);
}
void pose_H_10(double *state, double *unused, double *out_8672064356962455792) {
  H_10(state, unused, out_8672064356962455792);
}
void pose_h_13(double *state, double *unused, double *out_7526180503868782014) {
  h_13(state, unused, out_7526180503868782014);
}
void pose_H_13(double *state, double *unused, double *out_527469749874181770) {
  H_13(state, unused, out_527469749874181770);
}
void pose_h_14(double *state, double *unused, double *out_2452990038371875929) {
  h_14(state, unused, out_2452990038371875929);
}
void pose_H_14(double *state, double *unused, double *out_3119920602103034630) {
  H_14(state, unused, out_3119920602103034630);
}
void pose_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF pose = {
  .name = "pose",
  .kinds = { 4, 10, 13, 14 },
  .feature_kinds = {  },
  .f_fun = pose_f_fun,
  .F_fun = pose_F_fun,
  .err_fun = pose_err_fun,
  .inv_err_fun = pose_inv_err_fun,
  .H_mod_fun = pose_H_mod_fun,
  .predict = pose_predict,
  .hs = {
    { 4, pose_h_4 },
    { 10, pose_h_10 },
    { 13, pose_h_13 },
    { 14, pose_h_14 },
  },
  .Hs = {
    { 4, pose_H_4 },
    { 10, pose_H_10 },
    { 13, pose_H_13 },
    { 14, pose_H_14 },
  },
  .updates = {
    { 4, pose_update_4 },
    { 10, pose_update_10 },
    { 13, pose_update_13 },
    { 14, pose_update_14 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_lib_init(pose)
