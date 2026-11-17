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
void err_fun(double *nom_x, double *delta_x, double *out_8643847172382266479) {
   out_8643847172382266479[0] = delta_x[0] + nom_x[0];
   out_8643847172382266479[1] = delta_x[1] + nom_x[1];
   out_8643847172382266479[2] = delta_x[2] + nom_x[2];
   out_8643847172382266479[3] = delta_x[3] + nom_x[3];
   out_8643847172382266479[4] = delta_x[4] + nom_x[4];
   out_8643847172382266479[5] = delta_x[5] + nom_x[5];
   out_8643847172382266479[6] = delta_x[6] + nom_x[6];
   out_8643847172382266479[7] = delta_x[7] + nom_x[7];
   out_8643847172382266479[8] = delta_x[8] + nom_x[8];
   out_8643847172382266479[9] = delta_x[9] + nom_x[9];
   out_8643847172382266479[10] = delta_x[10] + nom_x[10];
   out_8643847172382266479[11] = delta_x[11] + nom_x[11];
   out_8643847172382266479[12] = delta_x[12] + nom_x[12];
   out_8643847172382266479[13] = delta_x[13] + nom_x[13];
   out_8643847172382266479[14] = delta_x[14] + nom_x[14];
   out_8643847172382266479[15] = delta_x[15] + nom_x[15];
   out_8643847172382266479[16] = delta_x[16] + nom_x[16];
   out_8643847172382266479[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1016762889049648120) {
   out_1016762889049648120[0] = -nom_x[0] + true_x[0];
   out_1016762889049648120[1] = -nom_x[1] + true_x[1];
   out_1016762889049648120[2] = -nom_x[2] + true_x[2];
   out_1016762889049648120[3] = -nom_x[3] + true_x[3];
   out_1016762889049648120[4] = -nom_x[4] + true_x[4];
   out_1016762889049648120[5] = -nom_x[5] + true_x[5];
   out_1016762889049648120[6] = -nom_x[6] + true_x[6];
   out_1016762889049648120[7] = -nom_x[7] + true_x[7];
   out_1016762889049648120[8] = -nom_x[8] + true_x[8];
   out_1016762889049648120[9] = -nom_x[9] + true_x[9];
   out_1016762889049648120[10] = -nom_x[10] + true_x[10];
   out_1016762889049648120[11] = -nom_x[11] + true_x[11];
   out_1016762889049648120[12] = -nom_x[12] + true_x[12];
   out_1016762889049648120[13] = -nom_x[13] + true_x[13];
   out_1016762889049648120[14] = -nom_x[14] + true_x[14];
   out_1016762889049648120[15] = -nom_x[15] + true_x[15];
   out_1016762889049648120[16] = -nom_x[16] + true_x[16];
   out_1016762889049648120[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_835593229450821917) {
   out_835593229450821917[0] = 1.0;
   out_835593229450821917[1] = 0.0;
   out_835593229450821917[2] = 0.0;
   out_835593229450821917[3] = 0.0;
   out_835593229450821917[4] = 0.0;
   out_835593229450821917[5] = 0.0;
   out_835593229450821917[6] = 0.0;
   out_835593229450821917[7] = 0.0;
   out_835593229450821917[8] = 0.0;
   out_835593229450821917[9] = 0.0;
   out_835593229450821917[10] = 0.0;
   out_835593229450821917[11] = 0.0;
   out_835593229450821917[12] = 0.0;
   out_835593229450821917[13] = 0.0;
   out_835593229450821917[14] = 0.0;
   out_835593229450821917[15] = 0.0;
   out_835593229450821917[16] = 0.0;
   out_835593229450821917[17] = 0.0;
   out_835593229450821917[18] = 0.0;
   out_835593229450821917[19] = 1.0;
   out_835593229450821917[20] = 0.0;
   out_835593229450821917[21] = 0.0;
   out_835593229450821917[22] = 0.0;
   out_835593229450821917[23] = 0.0;
   out_835593229450821917[24] = 0.0;
   out_835593229450821917[25] = 0.0;
   out_835593229450821917[26] = 0.0;
   out_835593229450821917[27] = 0.0;
   out_835593229450821917[28] = 0.0;
   out_835593229450821917[29] = 0.0;
   out_835593229450821917[30] = 0.0;
   out_835593229450821917[31] = 0.0;
   out_835593229450821917[32] = 0.0;
   out_835593229450821917[33] = 0.0;
   out_835593229450821917[34] = 0.0;
   out_835593229450821917[35] = 0.0;
   out_835593229450821917[36] = 0.0;
   out_835593229450821917[37] = 0.0;
   out_835593229450821917[38] = 1.0;
   out_835593229450821917[39] = 0.0;
   out_835593229450821917[40] = 0.0;
   out_835593229450821917[41] = 0.0;
   out_835593229450821917[42] = 0.0;
   out_835593229450821917[43] = 0.0;
   out_835593229450821917[44] = 0.0;
   out_835593229450821917[45] = 0.0;
   out_835593229450821917[46] = 0.0;
   out_835593229450821917[47] = 0.0;
   out_835593229450821917[48] = 0.0;
   out_835593229450821917[49] = 0.0;
   out_835593229450821917[50] = 0.0;
   out_835593229450821917[51] = 0.0;
   out_835593229450821917[52] = 0.0;
   out_835593229450821917[53] = 0.0;
   out_835593229450821917[54] = 0.0;
   out_835593229450821917[55] = 0.0;
   out_835593229450821917[56] = 0.0;
   out_835593229450821917[57] = 1.0;
   out_835593229450821917[58] = 0.0;
   out_835593229450821917[59] = 0.0;
   out_835593229450821917[60] = 0.0;
   out_835593229450821917[61] = 0.0;
   out_835593229450821917[62] = 0.0;
   out_835593229450821917[63] = 0.0;
   out_835593229450821917[64] = 0.0;
   out_835593229450821917[65] = 0.0;
   out_835593229450821917[66] = 0.0;
   out_835593229450821917[67] = 0.0;
   out_835593229450821917[68] = 0.0;
   out_835593229450821917[69] = 0.0;
   out_835593229450821917[70] = 0.0;
   out_835593229450821917[71] = 0.0;
   out_835593229450821917[72] = 0.0;
   out_835593229450821917[73] = 0.0;
   out_835593229450821917[74] = 0.0;
   out_835593229450821917[75] = 0.0;
   out_835593229450821917[76] = 1.0;
   out_835593229450821917[77] = 0.0;
   out_835593229450821917[78] = 0.0;
   out_835593229450821917[79] = 0.0;
   out_835593229450821917[80] = 0.0;
   out_835593229450821917[81] = 0.0;
   out_835593229450821917[82] = 0.0;
   out_835593229450821917[83] = 0.0;
   out_835593229450821917[84] = 0.0;
   out_835593229450821917[85] = 0.0;
   out_835593229450821917[86] = 0.0;
   out_835593229450821917[87] = 0.0;
   out_835593229450821917[88] = 0.0;
   out_835593229450821917[89] = 0.0;
   out_835593229450821917[90] = 0.0;
   out_835593229450821917[91] = 0.0;
   out_835593229450821917[92] = 0.0;
   out_835593229450821917[93] = 0.0;
   out_835593229450821917[94] = 0.0;
   out_835593229450821917[95] = 1.0;
   out_835593229450821917[96] = 0.0;
   out_835593229450821917[97] = 0.0;
   out_835593229450821917[98] = 0.0;
   out_835593229450821917[99] = 0.0;
   out_835593229450821917[100] = 0.0;
   out_835593229450821917[101] = 0.0;
   out_835593229450821917[102] = 0.0;
   out_835593229450821917[103] = 0.0;
   out_835593229450821917[104] = 0.0;
   out_835593229450821917[105] = 0.0;
   out_835593229450821917[106] = 0.0;
   out_835593229450821917[107] = 0.0;
   out_835593229450821917[108] = 0.0;
   out_835593229450821917[109] = 0.0;
   out_835593229450821917[110] = 0.0;
   out_835593229450821917[111] = 0.0;
   out_835593229450821917[112] = 0.0;
   out_835593229450821917[113] = 0.0;
   out_835593229450821917[114] = 1.0;
   out_835593229450821917[115] = 0.0;
   out_835593229450821917[116] = 0.0;
   out_835593229450821917[117] = 0.0;
   out_835593229450821917[118] = 0.0;
   out_835593229450821917[119] = 0.0;
   out_835593229450821917[120] = 0.0;
   out_835593229450821917[121] = 0.0;
   out_835593229450821917[122] = 0.0;
   out_835593229450821917[123] = 0.0;
   out_835593229450821917[124] = 0.0;
   out_835593229450821917[125] = 0.0;
   out_835593229450821917[126] = 0.0;
   out_835593229450821917[127] = 0.0;
   out_835593229450821917[128] = 0.0;
   out_835593229450821917[129] = 0.0;
   out_835593229450821917[130] = 0.0;
   out_835593229450821917[131] = 0.0;
   out_835593229450821917[132] = 0.0;
   out_835593229450821917[133] = 1.0;
   out_835593229450821917[134] = 0.0;
   out_835593229450821917[135] = 0.0;
   out_835593229450821917[136] = 0.0;
   out_835593229450821917[137] = 0.0;
   out_835593229450821917[138] = 0.0;
   out_835593229450821917[139] = 0.0;
   out_835593229450821917[140] = 0.0;
   out_835593229450821917[141] = 0.0;
   out_835593229450821917[142] = 0.0;
   out_835593229450821917[143] = 0.0;
   out_835593229450821917[144] = 0.0;
   out_835593229450821917[145] = 0.0;
   out_835593229450821917[146] = 0.0;
   out_835593229450821917[147] = 0.0;
   out_835593229450821917[148] = 0.0;
   out_835593229450821917[149] = 0.0;
   out_835593229450821917[150] = 0.0;
   out_835593229450821917[151] = 0.0;
   out_835593229450821917[152] = 1.0;
   out_835593229450821917[153] = 0.0;
   out_835593229450821917[154] = 0.0;
   out_835593229450821917[155] = 0.0;
   out_835593229450821917[156] = 0.0;
   out_835593229450821917[157] = 0.0;
   out_835593229450821917[158] = 0.0;
   out_835593229450821917[159] = 0.0;
   out_835593229450821917[160] = 0.0;
   out_835593229450821917[161] = 0.0;
   out_835593229450821917[162] = 0.0;
   out_835593229450821917[163] = 0.0;
   out_835593229450821917[164] = 0.0;
   out_835593229450821917[165] = 0.0;
   out_835593229450821917[166] = 0.0;
   out_835593229450821917[167] = 0.0;
   out_835593229450821917[168] = 0.0;
   out_835593229450821917[169] = 0.0;
   out_835593229450821917[170] = 0.0;
   out_835593229450821917[171] = 1.0;
   out_835593229450821917[172] = 0.0;
   out_835593229450821917[173] = 0.0;
   out_835593229450821917[174] = 0.0;
   out_835593229450821917[175] = 0.0;
   out_835593229450821917[176] = 0.0;
   out_835593229450821917[177] = 0.0;
   out_835593229450821917[178] = 0.0;
   out_835593229450821917[179] = 0.0;
   out_835593229450821917[180] = 0.0;
   out_835593229450821917[181] = 0.0;
   out_835593229450821917[182] = 0.0;
   out_835593229450821917[183] = 0.0;
   out_835593229450821917[184] = 0.0;
   out_835593229450821917[185] = 0.0;
   out_835593229450821917[186] = 0.0;
   out_835593229450821917[187] = 0.0;
   out_835593229450821917[188] = 0.0;
   out_835593229450821917[189] = 0.0;
   out_835593229450821917[190] = 1.0;
   out_835593229450821917[191] = 0.0;
   out_835593229450821917[192] = 0.0;
   out_835593229450821917[193] = 0.0;
   out_835593229450821917[194] = 0.0;
   out_835593229450821917[195] = 0.0;
   out_835593229450821917[196] = 0.0;
   out_835593229450821917[197] = 0.0;
   out_835593229450821917[198] = 0.0;
   out_835593229450821917[199] = 0.0;
   out_835593229450821917[200] = 0.0;
   out_835593229450821917[201] = 0.0;
   out_835593229450821917[202] = 0.0;
   out_835593229450821917[203] = 0.0;
   out_835593229450821917[204] = 0.0;
   out_835593229450821917[205] = 0.0;
   out_835593229450821917[206] = 0.0;
   out_835593229450821917[207] = 0.0;
   out_835593229450821917[208] = 0.0;
   out_835593229450821917[209] = 1.0;
   out_835593229450821917[210] = 0.0;
   out_835593229450821917[211] = 0.0;
   out_835593229450821917[212] = 0.0;
   out_835593229450821917[213] = 0.0;
   out_835593229450821917[214] = 0.0;
   out_835593229450821917[215] = 0.0;
   out_835593229450821917[216] = 0.0;
   out_835593229450821917[217] = 0.0;
   out_835593229450821917[218] = 0.0;
   out_835593229450821917[219] = 0.0;
   out_835593229450821917[220] = 0.0;
   out_835593229450821917[221] = 0.0;
   out_835593229450821917[222] = 0.0;
   out_835593229450821917[223] = 0.0;
   out_835593229450821917[224] = 0.0;
   out_835593229450821917[225] = 0.0;
   out_835593229450821917[226] = 0.0;
   out_835593229450821917[227] = 0.0;
   out_835593229450821917[228] = 1.0;
   out_835593229450821917[229] = 0.0;
   out_835593229450821917[230] = 0.0;
   out_835593229450821917[231] = 0.0;
   out_835593229450821917[232] = 0.0;
   out_835593229450821917[233] = 0.0;
   out_835593229450821917[234] = 0.0;
   out_835593229450821917[235] = 0.0;
   out_835593229450821917[236] = 0.0;
   out_835593229450821917[237] = 0.0;
   out_835593229450821917[238] = 0.0;
   out_835593229450821917[239] = 0.0;
   out_835593229450821917[240] = 0.0;
   out_835593229450821917[241] = 0.0;
   out_835593229450821917[242] = 0.0;
   out_835593229450821917[243] = 0.0;
   out_835593229450821917[244] = 0.0;
   out_835593229450821917[245] = 0.0;
   out_835593229450821917[246] = 0.0;
   out_835593229450821917[247] = 1.0;
   out_835593229450821917[248] = 0.0;
   out_835593229450821917[249] = 0.0;
   out_835593229450821917[250] = 0.0;
   out_835593229450821917[251] = 0.0;
   out_835593229450821917[252] = 0.0;
   out_835593229450821917[253] = 0.0;
   out_835593229450821917[254] = 0.0;
   out_835593229450821917[255] = 0.0;
   out_835593229450821917[256] = 0.0;
   out_835593229450821917[257] = 0.0;
   out_835593229450821917[258] = 0.0;
   out_835593229450821917[259] = 0.0;
   out_835593229450821917[260] = 0.0;
   out_835593229450821917[261] = 0.0;
   out_835593229450821917[262] = 0.0;
   out_835593229450821917[263] = 0.0;
   out_835593229450821917[264] = 0.0;
   out_835593229450821917[265] = 0.0;
   out_835593229450821917[266] = 1.0;
   out_835593229450821917[267] = 0.0;
   out_835593229450821917[268] = 0.0;
   out_835593229450821917[269] = 0.0;
   out_835593229450821917[270] = 0.0;
   out_835593229450821917[271] = 0.0;
   out_835593229450821917[272] = 0.0;
   out_835593229450821917[273] = 0.0;
   out_835593229450821917[274] = 0.0;
   out_835593229450821917[275] = 0.0;
   out_835593229450821917[276] = 0.0;
   out_835593229450821917[277] = 0.0;
   out_835593229450821917[278] = 0.0;
   out_835593229450821917[279] = 0.0;
   out_835593229450821917[280] = 0.0;
   out_835593229450821917[281] = 0.0;
   out_835593229450821917[282] = 0.0;
   out_835593229450821917[283] = 0.0;
   out_835593229450821917[284] = 0.0;
   out_835593229450821917[285] = 1.0;
   out_835593229450821917[286] = 0.0;
   out_835593229450821917[287] = 0.0;
   out_835593229450821917[288] = 0.0;
   out_835593229450821917[289] = 0.0;
   out_835593229450821917[290] = 0.0;
   out_835593229450821917[291] = 0.0;
   out_835593229450821917[292] = 0.0;
   out_835593229450821917[293] = 0.0;
   out_835593229450821917[294] = 0.0;
   out_835593229450821917[295] = 0.0;
   out_835593229450821917[296] = 0.0;
   out_835593229450821917[297] = 0.0;
   out_835593229450821917[298] = 0.0;
   out_835593229450821917[299] = 0.0;
   out_835593229450821917[300] = 0.0;
   out_835593229450821917[301] = 0.0;
   out_835593229450821917[302] = 0.0;
   out_835593229450821917[303] = 0.0;
   out_835593229450821917[304] = 1.0;
   out_835593229450821917[305] = 0.0;
   out_835593229450821917[306] = 0.0;
   out_835593229450821917[307] = 0.0;
   out_835593229450821917[308] = 0.0;
   out_835593229450821917[309] = 0.0;
   out_835593229450821917[310] = 0.0;
   out_835593229450821917[311] = 0.0;
   out_835593229450821917[312] = 0.0;
   out_835593229450821917[313] = 0.0;
   out_835593229450821917[314] = 0.0;
   out_835593229450821917[315] = 0.0;
   out_835593229450821917[316] = 0.0;
   out_835593229450821917[317] = 0.0;
   out_835593229450821917[318] = 0.0;
   out_835593229450821917[319] = 0.0;
   out_835593229450821917[320] = 0.0;
   out_835593229450821917[321] = 0.0;
   out_835593229450821917[322] = 0.0;
   out_835593229450821917[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_3336150215992616844) {
   out_3336150215992616844[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_3336150215992616844[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_3336150215992616844[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_3336150215992616844[3] = dt*state[12] + state[3];
   out_3336150215992616844[4] = dt*state[13] + state[4];
   out_3336150215992616844[5] = dt*state[14] + state[5];
   out_3336150215992616844[6] = state[6];
   out_3336150215992616844[7] = state[7];
   out_3336150215992616844[8] = state[8];
   out_3336150215992616844[9] = state[9];
   out_3336150215992616844[10] = state[10];
   out_3336150215992616844[11] = state[11];
   out_3336150215992616844[12] = state[12];
   out_3336150215992616844[13] = state[13];
   out_3336150215992616844[14] = state[14];
   out_3336150215992616844[15] = state[15];
   out_3336150215992616844[16] = state[16];
   out_3336150215992616844[17] = state[17];
}
void F_fun(double *state, double dt, double *out_8403437658028834064) {
   out_8403437658028834064[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8403437658028834064[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8403437658028834064[2] = 0;
   out_8403437658028834064[3] = 0;
   out_8403437658028834064[4] = 0;
   out_8403437658028834064[5] = 0;
   out_8403437658028834064[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8403437658028834064[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8403437658028834064[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_8403437658028834064[9] = 0;
   out_8403437658028834064[10] = 0;
   out_8403437658028834064[11] = 0;
   out_8403437658028834064[12] = 0;
   out_8403437658028834064[13] = 0;
   out_8403437658028834064[14] = 0;
   out_8403437658028834064[15] = 0;
   out_8403437658028834064[16] = 0;
   out_8403437658028834064[17] = 0;
   out_8403437658028834064[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8403437658028834064[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8403437658028834064[20] = 0;
   out_8403437658028834064[21] = 0;
   out_8403437658028834064[22] = 0;
   out_8403437658028834064[23] = 0;
   out_8403437658028834064[24] = 0;
   out_8403437658028834064[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8403437658028834064[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_8403437658028834064[27] = 0;
   out_8403437658028834064[28] = 0;
   out_8403437658028834064[29] = 0;
   out_8403437658028834064[30] = 0;
   out_8403437658028834064[31] = 0;
   out_8403437658028834064[32] = 0;
   out_8403437658028834064[33] = 0;
   out_8403437658028834064[34] = 0;
   out_8403437658028834064[35] = 0;
   out_8403437658028834064[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8403437658028834064[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8403437658028834064[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8403437658028834064[39] = 0;
   out_8403437658028834064[40] = 0;
   out_8403437658028834064[41] = 0;
   out_8403437658028834064[42] = 0;
   out_8403437658028834064[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8403437658028834064[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_8403437658028834064[45] = 0;
   out_8403437658028834064[46] = 0;
   out_8403437658028834064[47] = 0;
   out_8403437658028834064[48] = 0;
   out_8403437658028834064[49] = 0;
   out_8403437658028834064[50] = 0;
   out_8403437658028834064[51] = 0;
   out_8403437658028834064[52] = 0;
   out_8403437658028834064[53] = 0;
   out_8403437658028834064[54] = 0;
   out_8403437658028834064[55] = 0;
   out_8403437658028834064[56] = 0;
   out_8403437658028834064[57] = 1;
   out_8403437658028834064[58] = 0;
   out_8403437658028834064[59] = 0;
   out_8403437658028834064[60] = 0;
   out_8403437658028834064[61] = 0;
   out_8403437658028834064[62] = 0;
   out_8403437658028834064[63] = 0;
   out_8403437658028834064[64] = 0;
   out_8403437658028834064[65] = 0;
   out_8403437658028834064[66] = dt;
   out_8403437658028834064[67] = 0;
   out_8403437658028834064[68] = 0;
   out_8403437658028834064[69] = 0;
   out_8403437658028834064[70] = 0;
   out_8403437658028834064[71] = 0;
   out_8403437658028834064[72] = 0;
   out_8403437658028834064[73] = 0;
   out_8403437658028834064[74] = 0;
   out_8403437658028834064[75] = 0;
   out_8403437658028834064[76] = 1;
   out_8403437658028834064[77] = 0;
   out_8403437658028834064[78] = 0;
   out_8403437658028834064[79] = 0;
   out_8403437658028834064[80] = 0;
   out_8403437658028834064[81] = 0;
   out_8403437658028834064[82] = 0;
   out_8403437658028834064[83] = 0;
   out_8403437658028834064[84] = 0;
   out_8403437658028834064[85] = dt;
   out_8403437658028834064[86] = 0;
   out_8403437658028834064[87] = 0;
   out_8403437658028834064[88] = 0;
   out_8403437658028834064[89] = 0;
   out_8403437658028834064[90] = 0;
   out_8403437658028834064[91] = 0;
   out_8403437658028834064[92] = 0;
   out_8403437658028834064[93] = 0;
   out_8403437658028834064[94] = 0;
   out_8403437658028834064[95] = 1;
   out_8403437658028834064[96] = 0;
   out_8403437658028834064[97] = 0;
   out_8403437658028834064[98] = 0;
   out_8403437658028834064[99] = 0;
   out_8403437658028834064[100] = 0;
   out_8403437658028834064[101] = 0;
   out_8403437658028834064[102] = 0;
   out_8403437658028834064[103] = 0;
   out_8403437658028834064[104] = dt;
   out_8403437658028834064[105] = 0;
   out_8403437658028834064[106] = 0;
   out_8403437658028834064[107] = 0;
   out_8403437658028834064[108] = 0;
   out_8403437658028834064[109] = 0;
   out_8403437658028834064[110] = 0;
   out_8403437658028834064[111] = 0;
   out_8403437658028834064[112] = 0;
   out_8403437658028834064[113] = 0;
   out_8403437658028834064[114] = 1;
   out_8403437658028834064[115] = 0;
   out_8403437658028834064[116] = 0;
   out_8403437658028834064[117] = 0;
   out_8403437658028834064[118] = 0;
   out_8403437658028834064[119] = 0;
   out_8403437658028834064[120] = 0;
   out_8403437658028834064[121] = 0;
   out_8403437658028834064[122] = 0;
   out_8403437658028834064[123] = 0;
   out_8403437658028834064[124] = 0;
   out_8403437658028834064[125] = 0;
   out_8403437658028834064[126] = 0;
   out_8403437658028834064[127] = 0;
   out_8403437658028834064[128] = 0;
   out_8403437658028834064[129] = 0;
   out_8403437658028834064[130] = 0;
   out_8403437658028834064[131] = 0;
   out_8403437658028834064[132] = 0;
   out_8403437658028834064[133] = 1;
   out_8403437658028834064[134] = 0;
   out_8403437658028834064[135] = 0;
   out_8403437658028834064[136] = 0;
   out_8403437658028834064[137] = 0;
   out_8403437658028834064[138] = 0;
   out_8403437658028834064[139] = 0;
   out_8403437658028834064[140] = 0;
   out_8403437658028834064[141] = 0;
   out_8403437658028834064[142] = 0;
   out_8403437658028834064[143] = 0;
   out_8403437658028834064[144] = 0;
   out_8403437658028834064[145] = 0;
   out_8403437658028834064[146] = 0;
   out_8403437658028834064[147] = 0;
   out_8403437658028834064[148] = 0;
   out_8403437658028834064[149] = 0;
   out_8403437658028834064[150] = 0;
   out_8403437658028834064[151] = 0;
   out_8403437658028834064[152] = 1;
   out_8403437658028834064[153] = 0;
   out_8403437658028834064[154] = 0;
   out_8403437658028834064[155] = 0;
   out_8403437658028834064[156] = 0;
   out_8403437658028834064[157] = 0;
   out_8403437658028834064[158] = 0;
   out_8403437658028834064[159] = 0;
   out_8403437658028834064[160] = 0;
   out_8403437658028834064[161] = 0;
   out_8403437658028834064[162] = 0;
   out_8403437658028834064[163] = 0;
   out_8403437658028834064[164] = 0;
   out_8403437658028834064[165] = 0;
   out_8403437658028834064[166] = 0;
   out_8403437658028834064[167] = 0;
   out_8403437658028834064[168] = 0;
   out_8403437658028834064[169] = 0;
   out_8403437658028834064[170] = 0;
   out_8403437658028834064[171] = 1;
   out_8403437658028834064[172] = 0;
   out_8403437658028834064[173] = 0;
   out_8403437658028834064[174] = 0;
   out_8403437658028834064[175] = 0;
   out_8403437658028834064[176] = 0;
   out_8403437658028834064[177] = 0;
   out_8403437658028834064[178] = 0;
   out_8403437658028834064[179] = 0;
   out_8403437658028834064[180] = 0;
   out_8403437658028834064[181] = 0;
   out_8403437658028834064[182] = 0;
   out_8403437658028834064[183] = 0;
   out_8403437658028834064[184] = 0;
   out_8403437658028834064[185] = 0;
   out_8403437658028834064[186] = 0;
   out_8403437658028834064[187] = 0;
   out_8403437658028834064[188] = 0;
   out_8403437658028834064[189] = 0;
   out_8403437658028834064[190] = 1;
   out_8403437658028834064[191] = 0;
   out_8403437658028834064[192] = 0;
   out_8403437658028834064[193] = 0;
   out_8403437658028834064[194] = 0;
   out_8403437658028834064[195] = 0;
   out_8403437658028834064[196] = 0;
   out_8403437658028834064[197] = 0;
   out_8403437658028834064[198] = 0;
   out_8403437658028834064[199] = 0;
   out_8403437658028834064[200] = 0;
   out_8403437658028834064[201] = 0;
   out_8403437658028834064[202] = 0;
   out_8403437658028834064[203] = 0;
   out_8403437658028834064[204] = 0;
   out_8403437658028834064[205] = 0;
   out_8403437658028834064[206] = 0;
   out_8403437658028834064[207] = 0;
   out_8403437658028834064[208] = 0;
   out_8403437658028834064[209] = 1;
   out_8403437658028834064[210] = 0;
   out_8403437658028834064[211] = 0;
   out_8403437658028834064[212] = 0;
   out_8403437658028834064[213] = 0;
   out_8403437658028834064[214] = 0;
   out_8403437658028834064[215] = 0;
   out_8403437658028834064[216] = 0;
   out_8403437658028834064[217] = 0;
   out_8403437658028834064[218] = 0;
   out_8403437658028834064[219] = 0;
   out_8403437658028834064[220] = 0;
   out_8403437658028834064[221] = 0;
   out_8403437658028834064[222] = 0;
   out_8403437658028834064[223] = 0;
   out_8403437658028834064[224] = 0;
   out_8403437658028834064[225] = 0;
   out_8403437658028834064[226] = 0;
   out_8403437658028834064[227] = 0;
   out_8403437658028834064[228] = 1;
   out_8403437658028834064[229] = 0;
   out_8403437658028834064[230] = 0;
   out_8403437658028834064[231] = 0;
   out_8403437658028834064[232] = 0;
   out_8403437658028834064[233] = 0;
   out_8403437658028834064[234] = 0;
   out_8403437658028834064[235] = 0;
   out_8403437658028834064[236] = 0;
   out_8403437658028834064[237] = 0;
   out_8403437658028834064[238] = 0;
   out_8403437658028834064[239] = 0;
   out_8403437658028834064[240] = 0;
   out_8403437658028834064[241] = 0;
   out_8403437658028834064[242] = 0;
   out_8403437658028834064[243] = 0;
   out_8403437658028834064[244] = 0;
   out_8403437658028834064[245] = 0;
   out_8403437658028834064[246] = 0;
   out_8403437658028834064[247] = 1;
   out_8403437658028834064[248] = 0;
   out_8403437658028834064[249] = 0;
   out_8403437658028834064[250] = 0;
   out_8403437658028834064[251] = 0;
   out_8403437658028834064[252] = 0;
   out_8403437658028834064[253] = 0;
   out_8403437658028834064[254] = 0;
   out_8403437658028834064[255] = 0;
   out_8403437658028834064[256] = 0;
   out_8403437658028834064[257] = 0;
   out_8403437658028834064[258] = 0;
   out_8403437658028834064[259] = 0;
   out_8403437658028834064[260] = 0;
   out_8403437658028834064[261] = 0;
   out_8403437658028834064[262] = 0;
   out_8403437658028834064[263] = 0;
   out_8403437658028834064[264] = 0;
   out_8403437658028834064[265] = 0;
   out_8403437658028834064[266] = 1;
   out_8403437658028834064[267] = 0;
   out_8403437658028834064[268] = 0;
   out_8403437658028834064[269] = 0;
   out_8403437658028834064[270] = 0;
   out_8403437658028834064[271] = 0;
   out_8403437658028834064[272] = 0;
   out_8403437658028834064[273] = 0;
   out_8403437658028834064[274] = 0;
   out_8403437658028834064[275] = 0;
   out_8403437658028834064[276] = 0;
   out_8403437658028834064[277] = 0;
   out_8403437658028834064[278] = 0;
   out_8403437658028834064[279] = 0;
   out_8403437658028834064[280] = 0;
   out_8403437658028834064[281] = 0;
   out_8403437658028834064[282] = 0;
   out_8403437658028834064[283] = 0;
   out_8403437658028834064[284] = 0;
   out_8403437658028834064[285] = 1;
   out_8403437658028834064[286] = 0;
   out_8403437658028834064[287] = 0;
   out_8403437658028834064[288] = 0;
   out_8403437658028834064[289] = 0;
   out_8403437658028834064[290] = 0;
   out_8403437658028834064[291] = 0;
   out_8403437658028834064[292] = 0;
   out_8403437658028834064[293] = 0;
   out_8403437658028834064[294] = 0;
   out_8403437658028834064[295] = 0;
   out_8403437658028834064[296] = 0;
   out_8403437658028834064[297] = 0;
   out_8403437658028834064[298] = 0;
   out_8403437658028834064[299] = 0;
   out_8403437658028834064[300] = 0;
   out_8403437658028834064[301] = 0;
   out_8403437658028834064[302] = 0;
   out_8403437658028834064[303] = 0;
   out_8403437658028834064[304] = 1;
   out_8403437658028834064[305] = 0;
   out_8403437658028834064[306] = 0;
   out_8403437658028834064[307] = 0;
   out_8403437658028834064[308] = 0;
   out_8403437658028834064[309] = 0;
   out_8403437658028834064[310] = 0;
   out_8403437658028834064[311] = 0;
   out_8403437658028834064[312] = 0;
   out_8403437658028834064[313] = 0;
   out_8403437658028834064[314] = 0;
   out_8403437658028834064[315] = 0;
   out_8403437658028834064[316] = 0;
   out_8403437658028834064[317] = 0;
   out_8403437658028834064[318] = 0;
   out_8403437658028834064[319] = 0;
   out_8403437658028834064[320] = 0;
   out_8403437658028834064[321] = 0;
   out_8403437658028834064[322] = 0;
   out_8403437658028834064[323] = 1;
}
void h_4(double *state, double *unused, double *out_4224362500834756736) {
   out_4224362500834756736[0] = state[6] + state[9];
   out_4224362500834756736[1] = state[7] + state[10];
   out_4224362500834756736[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_4036003130552291941) {
   out_4036003130552291941[0] = 0;
   out_4036003130552291941[1] = 0;
   out_4036003130552291941[2] = 0;
   out_4036003130552291941[3] = 0;
   out_4036003130552291941[4] = 0;
   out_4036003130552291941[5] = 0;
   out_4036003130552291941[6] = 1;
   out_4036003130552291941[7] = 0;
   out_4036003130552291941[8] = 0;
   out_4036003130552291941[9] = 1;
   out_4036003130552291941[10] = 0;
   out_4036003130552291941[11] = 0;
   out_4036003130552291941[12] = 0;
   out_4036003130552291941[13] = 0;
   out_4036003130552291941[14] = 0;
   out_4036003130552291941[15] = 0;
   out_4036003130552291941[16] = 0;
   out_4036003130552291941[17] = 0;
   out_4036003130552291941[18] = 0;
   out_4036003130552291941[19] = 0;
   out_4036003130552291941[20] = 0;
   out_4036003130552291941[21] = 0;
   out_4036003130552291941[22] = 0;
   out_4036003130552291941[23] = 0;
   out_4036003130552291941[24] = 0;
   out_4036003130552291941[25] = 1;
   out_4036003130552291941[26] = 0;
   out_4036003130552291941[27] = 0;
   out_4036003130552291941[28] = 1;
   out_4036003130552291941[29] = 0;
   out_4036003130552291941[30] = 0;
   out_4036003130552291941[31] = 0;
   out_4036003130552291941[32] = 0;
   out_4036003130552291941[33] = 0;
   out_4036003130552291941[34] = 0;
   out_4036003130552291941[35] = 0;
   out_4036003130552291941[36] = 0;
   out_4036003130552291941[37] = 0;
   out_4036003130552291941[38] = 0;
   out_4036003130552291941[39] = 0;
   out_4036003130552291941[40] = 0;
   out_4036003130552291941[41] = 0;
   out_4036003130552291941[42] = 0;
   out_4036003130552291941[43] = 0;
   out_4036003130552291941[44] = 1;
   out_4036003130552291941[45] = 0;
   out_4036003130552291941[46] = 0;
   out_4036003130552291941[47] = 1;
   out_4036003130552291941[48] = 0;
   out_4036003130552291941[49] = 0;
   out_4036003130552291941[50] = 0;
   out_4036003130552291941[51] = 0;
   out_4036003130552291941[52] = 0;
   out_4036003130552291941[53] = 0;
}
void h_10(double *state, double *unused, double *out_7916371707404069928) {
   out_7916371707404069928[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_7916371707404069928[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_7916371707404069928[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_2905842532402009938) {
   out_2905842532402009938[0] = 0;
   out_2905842532402009938[1] = 9.8100000000000005*cos(state[1]);
   out_2905842532402009938[2] = 0;
   out_2905842532402009938[3] = 0;
   out_2905842532402009938[4] = -state[8];
   out_2905842532402009938[5] = state[7];
   out_2905842532402009938[6] = 0;
   out_2905842532402009938[7] = state[5];
   out_2905842532402009938[8] = -state[4];
   out_2905842532402009938[9] = 0;
   out_2905842532402009938[10] = 0;
   out_2905842532402009938[11] = 0;
   out_2905842532402009938[12] = 1;
   out_2905842532402009938[13] = 0;
   out_2905842532402009938[14] = 0;
   out_2905842532402009938[15] = 1;
   out_2905842532402009938[16] = 0;
   out_2905842532402009938[17] = 0;
   out_2905842532402009938[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_2905842532402009938[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_2905842532402009938[20] = 0;
   out_2905842532402009938[21] = state[8];
   out_2905842532402009938[22] = 0;
   out_2905842532402009938[23] = -state[6];
   out_2905842532402009938[24] = -state[5];
   out_2905842532402009938[25] = 0;
   out_2905842532402009938[26] = state[3];
   out_2905842532402009938[27] = 0;
   out_2905842532402009938[28] = 0;
   out_2905842532402009938[29] = 0;
   out_2905842532402009938[30] = 0;
   out_2905842532402009938[31] = 1;
   out_2905842532402009938[32] = 0;
   out_2905842532402009938[33] = 0;
   out_2905842532402009938[34] = 1;
   out_2905842532402009938[35] = 0;
   out_2905842532402009938[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_2905842532402009938[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_2905842532402009938[38] = 0;
   out_2905842532402009938[39] = -state[7];
   out_2905842532402009938[40] = state[6];
   out_2905842532402009938[41] = 0;
   out_2905842532402009938[42] = state[4];
   out_2905842532402009938[43] = -state[3];
   out_2905842532402009938[44] = 0;
   out_2905842532402009938[45] = 0;
   out_2905842532402009938[46] = 0;
   out_2905842532402009938[47] = 0;
   out_2905842532402009938[48] = 0;
   out_2905842532402009938[49] = 0;
   out_2905842532402009938[50] = 1;
   out_2905842532402009938[51] = 0;
   out_2905842532402009938[52] = 0;
   out_2905842532402009938[53] = 1;
}
void h_13(double *state, double *unused, double *out_1381557700253186812) {
   out_1381557700253186812[0] = state[3];
   out_1381557700253186812[1] = state[4];
   out_1381557700253186812[2] = state[5];
}
void H_13(double *state, double *unused, double *out_3574628077764408988) {
   out_3574628077764408988[0] = 0;
   out_3574628077764408988[1] = 0;
   out_3574628077764408988[2] = 0;
   out_3574628077764408988[3] = 1;
   out_3574628077764408988[4] = 0;
   out_3574628077764408988[5] = 0;
   out_3574628077764408988[6] = 0;
   out_3574628077764408988[7] = 0;
   out_3574628077764408988[8] = 0;
   out_3574628077764408988[9] = 0;
   out_3574628077764408988[10] = 0;
   out_3574628077764408988[11] = 0;
   out_3574628077764408988[12] = 0;
   out_3574628077764408988[13] = 0;
   out_3574628077764408988[14] = 0;
   out_3574628077764408988[15] = 0;
   out_3574628077764408988[16] = 0;
   out_3574628077764408988[17] = 0;
   out_3574628077764408988[18] = 0;
   out_3574628077764408988[19] = 0;
   out_3574628077764408988[20] = 0;
   out_3574628077764408988[21] = 0;
   out_3574628077764408988[22] = 1;
   out_3574628077764408988[23] = 0;
   out_3574628077764408988[24] = 0;
   out_3574628077764408988[25] = 0;
   out_3574628077764408988[26] = 0;
   out_3574628077764408988[27] = 0;
   out_3574628077764408988[28] = 0;
   out_3574628077764408988[29] = 0;
   out_3574628077764408988[30] = 0;
   out_3574628077764408988[31] = 0;
   out_3574628077764408988[32] = 0;
   out_3574628077764408988[33] = 0;
   out_3574628077764408988[34] = 0;
   out_3574628077764408988[35] = 0;
   out_3574628077764408988[36] = 0;
   out_3574628077764408988[37] = 0;
   out_3574628077764408988[38] = 0;
   out_3574628077764408988[39] = 0;
   out_3574628077764408988[40] = 0;
   out_3574628077764408988[41] = 1;
   out_3574628077764408988[42] = 0;
   out_3574628077764408988[43] = 0;
   out_3574628077764408988[44] = 0;
   out_3574628077764408988[45] = 0;
   out_3574628077764408988[46] = 0;
   out_3574628077764408988[47] = 0;
   out_3574628077764408988[48] = 0;
   out_3574628077764408988[49] = 0;
   out_3574628077764408988[50] = 0;
   out_3574628077764408988[51] = 0;
   out_3574628077764408988[52] = 0;
   out_3574628077764408988[53] = 0;
}
void h_14(double *state, double *unused, double *out_7719697066185784283) {
   out_7719697066185784283[0] = state[6];
   out_7719697066185784283[1] = state[7];
   out_7719697066185784283[2] = state[8];
}
void H_14(double *state, double *unused, double *out_7118791562847664237) {
   out_7118791562847664237[0] = 0;
   out_7118791562847664237[1] = 0;
   out_7118791562847664237[2] = 0;
   out_7118791562847664237[3] = 0;
   out_7118791562847664237[4] = 0;
   out_7118791562847664237[5] = 0;
   out_7118791562847664237[6] = 1;
   out_7118791562847664237[7] = 0;
   out_7118791562847664237[8] = 0;
   out_7118791562847664237[9] = 0;
   out_7118791562847664237[10] = 0;
   out_7118791562847664237[11] = 0;
   out_7118791562847664237[12] = 0;
   out_7118791562847664237[13] = 0;
   out_7118791562847664237[14] = 0;
   out_7118791562847664237[15] = 0;
   out_7118791562847664237[16] = 0;
   out_7118791562847664237[17] = 0;
   out_7118791562847664237[18] = 0;
   out_7118791562847664237[19] = 0;
   out_7118791562847664237[20] = 0;
   out_7118791562847664237[21] = 0;
   out_7118791562847664237[22] = 0;
   out_7118791562847664237[23] = 0;
   out_7118791562847664237[24] = 0;
   out_7118791562847664237[25] = 1;
   out_7118791562847664237[26] = 0;
   out_7118791562847664237[27] = 0;
   out_7118791562847664237[28] = 0;
   out_7118791562847664237[29] = 0;
   out_7118791562847664237[30] = 0;
   out_7118791562847664237[31] = 0;
   out_7118791562847664237[32] = 0;
   out_7118791562847664237[33] = 0;
   out_7118791562847664237[34] = 0;
   out_7118791562847664237[35] = 0;
   out_7118791562847664237[36] = 0;
   out_7118791562847664237[37] = 0;
   out_7118791562847664237[38] = 0;
   out_7118791562847664237[39] = 0;
   out_7118791562847664237[40] = 0;
   out_7118791562847664237[41] = 0;
   out_7118791562847664237[42] = 0;
   out_7118791562847664237[43] = 0;
   out_7118791562847664237[44] = 1;
   out_7118791562847664237[45] = 0;
   out_7118791562847664237[46] = 0;
   out_7118791562847664237[47] = 0;
   out_7118791562847664237[48] = 0;
   out_7118791562847664237[49] = 0;
   out_7118791562847664237[50] = 0;
   out_7118791562847664237[51] = 0;
   out_7118791562847664237[52] = 0;
   out_7118791562847664237[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_8643847172382266479) {
  err_fun(nom_x, delta_x, out_8643847172382266479);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1016762889049648120) {
  inv_err_fun(nom_x, true_x, out_1016762889049648120);
}
void pose_H_mod_fun(double *state, double *out_835593229450821917) {
  H_mod_fun(state, out_835593229450821917);
}
void pose_f_fun(double *state, double dt, double *out_3336150215992616844) {
  f_fun(state,  dt, out_3336150215992616844);
}
void pose_F_fun(double *state, double dt, double *out_8403437658028834064) {
  F_fun(state,  dt, out_8403437658028834064);
}
void pose_h_4(double *state, double *unused, double *out_4224362500834756736) {
  h_4(state, unused, out_4224362500834756736);
}
void pose_H_4(double *state, double *unused, double *out_4036003130552291941) {
  H_4(state, unused, out_4036003130552291941);
}
void pose_h_10(double *state, double *unused, double *out_7916371707404069928) {
  h_10(state, unused, out_7916371707404069928);
}
void pose_H_10(double *state, double *unused, double *out_2905842532402009938) {
  H_10(state, unused, out_2905842532402009938);
}
void pose_h_13(double *state, double *unused, double *out_1381557700253186812) {
  h_13(state, unused, out_1381557700253186812);
}
void pose_H_13(double *state, double *unused, double *out_3574628077764408988) {
  H_13(state, unused, out_3574628077764408988);
}
void pose_h_14(double *state, double *unused, double *out_7719697066185784283) {
  h_14(state, unused, out_7719697066185784283);
}
void pose_H_14(double *state, double *unused, double *out_7118791562847664237) {
  H_14(state, unused, out_7118791562847664237);
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
