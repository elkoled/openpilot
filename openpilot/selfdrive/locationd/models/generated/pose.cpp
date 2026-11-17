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
void err_fun(double *nom_x, double *delta_x, double *out_4215104848141748955) {
   out_4215104848141748955[0] = delta_x[0] + nom_x[0];
   out_4215104848141748955[1] = delta_x[1] + nom_x[1];
   out_4215104848141748955[2] = delta_x[2] + nom_x[2];
   out_4215104848141748955[3] = delta_x[3] + nom_x[3];
   out_4215104848141748955[4] = delta_x[4] + nom_x[4];
   out_4215104848141748955[5] = delta_x[5] + nom_x[5];
   out_4215104848141748955[6] = delta_x[6] + nom_x[6];
   out_4215104848141748955[7] = delta_x[7] + nom_x[7];
   out_4215104848141748955[8] = delta_x[8] + nom_x[8];
   out_4215104848141748955[9] = delta_x[9] + nom_x[9];
   out_4215104848141748955[10] = delta_x[10] + nom_x[10];
   out_4215104848141748955[11] = delta_x[11] + nom_x[11];
   out_4215104848141748955[12] = delta_x[12] + nom_x[12];
   out_4215104848141748955[13] = delta_x[13] + nom_x[13];
   out_4215104848141748955[14] = delta_x[14] + nom_x[14];
   out_4215104848141748955[15] = delta_x[15] + nom_x[15];
   out_4215104848141748955[16] = delta_x[16] + nom_x[16];
   out_4215104848141748955[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8045931564468893177) {
   out_8045931564468893177[0] = -nom_x[0] + true_x[0];
   out_8045931564468893177[1] = -nom_x[1] + true_x[1];
   out_8045931564468893177[2] = -nom_x[2] + true_x[2];
   out_8045931564468893177[3] = -nom_x[3] + true_x[3];
   out_8045931564468893177[4] = -nom_x[4] + true_x[4];
   out_8045931564468893177[5] = -nom_x[5] + true_x[5];
   out_8045931564468893177[6] = -nom_x[6] + true_x[6];
   out_8045931564468893177[7] = -nom_x[7] + true_x[7];
   out_8045931564468893177[8] = -nom_x[8] + true_x[8];
   out_8045931564468893177[9] = -nom_x[9] + true_x[9];
   out_8045931564468893177[10] = -nom_x[10] + true_x[10];
   out_8045931564468893177[11] = -nom_x[11] + true_x[11];
   out_8045931564468893177[12] = -nom_x[12] + true_x[12];
   out_8045931564468893177[13] = -nom_x[13] + true_x[13];
   out_8045931564468893177[14] = -nom_x[14] + true_x[14];
   out_8045931564468893177[15] = -nom_x[15] + true_x[15];
   out_8045931564468893177[16] = -nom_x[16] + true_x[16];
   out_8045931564468893177[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_6658005824480305271) {
   out_6658005824480305271[0] = 1.0;
   out_6658005824480305271[1] = 0.0;
   out_6658005824480305271[2] = 0.0;
   out_6658005824480305271[3] = 0.0;
   out_6658005824480305271[4] = 0.0;
   out_6658005824480305271[5] = 0.0;
   out_6658005824480305271[6] = 0.0;
   out_6658005824480305271[7] = 0.0;
   out_6658005824480305271[8] = 0.0;
   out_6658005824480305271[9] = 0.0;
   out_6658005824480305271[10] = 0.0;
   out_6658005824480305271[11] = 0.0;
   out_6658005824480305271[12] = 0.0;
   out_6658005824480305271[13] = 0.0;
   out_6658005824480305271[14] = 0.0;
   out_6658005824480305271[15] = 0.0;
   out_6658005824480305271[16] = 0.0;
   out_6658005824480305271[17] = 0.0;
   out_6658005824480305271[18] = 0.0;
   out_6658005824480305271[19] = 1.0;
   out_6658005824480305271[20] = 0.0;
   out_6658005824480305271[21] = 0.0;
   out_6658005824480305271[22] = 0.0;
   out_6658005824480305271[23] = 0.0;
   out_6658005824480305271[24] = 0.0;
   out_6658005824480305271[25] = 0.0;
   out_6658005824480305271[26] = 0.0;
   out_6658005824480305271[27] = 0.0;
   out_6658005824480305271[28] = 0.0;
   out_6658005824480305271[29] = 0.0;
   out_6658005824480305271[30] = 0.0;
   out_6658005824480305271[31] = 0.0;
   out_6658005824480305271[32] = 0.0;
   out_6658005824480305271[33] = 0.0;
   out_6658005824480305271[34] = 0.0;
   out_6658005824480305271[35] = 0.0;
   out_6658005824480305271[36] = 0.0;
   out_6658005824480305271[37] = 0.0;
   out_6658005824480305271[38] = 1.0;
   out_6658005824480305271[39] = 0.0;
   out_6658005824480305271[40] = 0.0;
   out_6658005824480305271[41] = 0.0;
   out_6658005824480305271[42] = 0.0;
   out_6658005824480305271[43] = 0.0;
   out_6658005824480305271[44] = 0.0;
   out_6658005824480305271[45] = 0.0;
   out_6658005824480305271[46] = 0.0;
   out_6658005824480305271[47] = 0.0;
   out_6658005824480305271[48] = 0.0;
   out_6658005824480305271[49] = 0.0;
   out_6658005824480305271[50] = 0.0;
   out_6658005824480305271[51] = 0.0;
   out_6658005824480305271[52] = 0.0;
   out_6658005824480305271[53] = 0.0;
   out_6658005824480305271[54] = 0.0;
   out_6658005824480305271[55] = 0.0;
   out_6658005824480305271[56] = 0.0;
   out_6658005824480305271[57] = 1.0;
   out_6658005824480305271[58] = 0.0;
   out_6658005824480305271[59] = 0.0;
   out_6658005824480305271[60] = 0.0;
   out_6658005824480305271[61] = 0.0;
   out_6658005824480305271[62] = 0.0;
   out_6658005824480305271[63] = 0.0;
   out_6658005824480305271[64] = 0.0;
   out_6658005824480305271[65] = 0.0;
   out_6658005824480305271[66] = 0.0;
   out_6658005824480305271[67] = 0.0;
   out_6658005824480305271[68] = 0.0;
   out_6658005824480305271[69] = 0.0;
   out_6658005824480305271[70] = 0.0;
   out_6658005824480305271[71] = 0.0;
   out_6658005824480305271[72] = 0.0;
   out_6658005824480305271[73] = 0.0;
   out_6658005824480305271[74] = 0.0;
   out_6658005824480305271[75] = 0.0;
   out_6658005824480305271[76] = 1.0;
   out_6658005824480305271[77] = 0.0;
   out_6658005824480305271[78] = 0.0;
   out_6658005824480305271[79] = 0.0;
   out_6658005824480305271[80] = 0.0;
   out_6658005824480305271[81] = 0.0;
   out_6658005824480305271[82] = 0.0;
   out_6658005824480305271[83] = 0.0;
   out_6658005824480305271[84] = 0.0;
   out_6658005824480305271[85] = 0.0;
   out_6658005824480305271[86] = 0.0;
   out_6658005824480305271[87] = 0.0;
   out_6658005824480305271[88] = 0.0;
   out_6658005824480305271[89] = 0.0;
   out_6658005824480305271[90] = 0.0;
   out_6658005824480305271[91] = 0.0;
   out_6658005824480305271[92] = 0.0;
   out_6658005824480305271[93] = 0.0;
   out_6658005824480305271[94] = 0.0;
   out_6658005824480305271[95] = 1.0;
   out_6658005824480305271[96] = 0.0;
   out_6658005824480305271[97] = 0.0;
   out_6658005824480305271[98] = 0.0;
   out_6658005824480305271[99] = 0.0;
   out_6658005824480305271[100] = 0.0;
   out_6658005824480305271[101] = 0.0;
   out_6658005824480305271[102] = 0.0;
   out_6658005824480305271[103] = 0.0;
   out_6658005824480305271[104] = 0.0;
   out_6658005824480305271[105] = 0.0;
   out_6658005824480305271[106] = 0.0;
   out_6658005824480305271[107] = 0.0;
   out_6658005824480305271[108] = 0.0;
   out_6658005824480305271[109] = 0.0;
   out_6658005824480305271[110] = 0.0;
   out_6658005824480305271[111] = 0.0;
   out_6658005824480305271[112] = 0.0;
   out_6658005824480305271[113] = 0.0;
   out_6658005824480305271[114] = 1.0;
   out_6658005824480305271[115] = 0.0;
   out_6658005824480305271[116] = 0.0;
   out_6658005824480305271[117] = 0.0;
   out_6658005824480305271[118] = 0.0;
   out_6658005824480305271[119] = 0.0;
   out_6658005824480305271[120] = 0.0;
   out_6658005824480305271[121] = 0.0;
   out_6658005824480305271[122] = 0.0;
   out_6658005824480305271[123] = 0.0;
   out_6658005824480305271[124] = 0.0;
   out_6658005824480305271[125] = 0.0;
   out_6658005824480305271[126] = 0.0;
   out_6658005824480305271[127] = 0.0;
   out_6658005824480305271[128] = 0.0;
   out_6658005824480305271[129] = 0.0;
   out_6658005824480305271[130] = 0.0;
   out_6658005824480305271[131] = 0.0;
   out_6658005824480305271[132] = 0.0;
   out_6658005824480305271[133] = 1.0;
   out_6658005824480305271[134] = 0.0;
   out_6658005824480305271[135] = 0.0;
   out_6658005824480305271[136] = 0.0;
   out_6658005824480305271[137] = 0.0;
   out_6658005824480305271[138] = 0.0;
   out_6658005824480305271[139] = 0.0;
   out_6658005824480305271[140] = 0.0;
   out_6658005824480305271[141] = 0.0;
   out_6658005824480305271[142] = 0.0;
   out_6658005824480305271[143] = 0.0;
   out_6658005824480305271[144] = 0.0;
   out_6658005824480305271[145] = 0.0;
   out_6658005824480305271[146] = 0.0;
   out_6658005824480305271[147] = 0.0;
   out_6658005824480305271[148] = 0.0;
   out_6658005824480305271[149] = 0.0;
   out_6658005824480305271[150] = 0.0;
   out_6658005824480305271[151] = 0.0;
   out_6658005824480305271[152] = 1.0;
   out_6658005824480305271[153] = 0.0;
   out_6658005824480305271[154] = 0.0;
   out_6658005824480305271[155] = 0.0;
   out_6658005824480305271[156] = 0.0;
   out_6658005824480305271[157] = 0.0;
   out_6658005824480305271[158] = 0.0;
   out_6658005824480305271[159] = 0.0;
   out_6658005824480305271[160] = 0.0;
   out_6658005824480305271[161] = 0.0;
   out_6658005824480305271[162] = 0.0;
   out_6658005824480305271[163] = 0.0;
   out_6658005824480305271[164] = 0.0;
   out_6658005824480305271[165] = 0.0;
   out_6658005824480305271[166] = 0.0;
   out_6658005824480305271[167] = 0.0;
   out_6658005824480305271[168] = 0.0;
   out_6658005824480305271[169] = 0.0;
   out_6658005824480305271[170] = 0.0;
   out_6658005824480305271[171] = 1.0;
   out_6658005824480305271[172] = 0.0;
   out_6658005824480305271[173] = 0.0;
   out_6658005824480305271[174] = 0.0;
   out_6658005824480305271[175] = 0.0;
   out_6658005824480305271[176] = 0.0;
   out_6658005824480305271[177] = 0.0;
   out_6658005824480305271[178] = 0.0;
   out_6658005824480305271[179] = 0.0;
   out_6658005824480305271[180] = 0.0;
   out_6658005824480305271[181] = 0.0;
   out_6658005824480305271[182] = 0.0;
   out_6658005824480305271[183] = 0.0;
   out_6658005824480305271[184] = 0.0;
   out_6658005824480305271[185] = 0.0;
   out_6658005824480305271[186] = 0.0;
   out_6658005824480305271[187] = 0.0;
   out_6658005824480305271[188] = 0.0;
   out_6658005824480305271[189] = 0.0;
   out_6658005824480305271[190] = 1.0;
   out_6658005824480305271[191] = 0.0;
   out_6658005824480305271[192] = 0.0;
   out_6658005824480305271[193] = 0.0;
   out_6658005824480305271[194] = 0.0;
   out_6658005824480305271[195] = 0.0;
   out_6658005824480305271[196] = 0.0;
   out_6658005824480305271[197] = 0.0;
   out_6658005824480305271[198] = 0.0;
   out_6658005824480305271[199] = 0.0;
   out_6658005824480305271[200] = 0.0;
   out_6658005824480305271[201] = 0.0;
   out_6658005824480305271[202] = 0.0;
   out_6658005824480305271[203] = 0.0;
   out_6658005824480305271[204] = 0.0;
   out_6658005824480305271[205] = 0.0;
   out_6658005824480305271[206] = 0.0;
   out_6658005824480305271[207] = 0.0;
   out_6658005824480305271[208] = 0.0;
   out_6658005824480305271[209] = 1.0;
   out_6658005824480305271[210] = 0.0;
   out_6658005824480305271[211] = 0.0;
   out_6658005824480305271[212] = 0.0;
   out_6658005824480305271[213] = 0.0;
   out_6658005824480305271[214] = 0.0;
   out_6658005824480305271[215] = 0.0;
   out_6658005824480305271[216] = 0.0;
   out_6658005824480305271[217] = 0.0;
   out_6658005824480305271[218] = 0.0;
   out_6658005824480305271[219] = 0.0;
   out_6658005824480305271[220] = 0.0;
   out_6658005824480305271[221] = 0.0;
   out_6658005824480305271[222] = 0.0;
   out_6658005824480305271[223] = 0.0;
   out_6658005824480305271[224] = 0.0;
   out_6658005824480305271[225] = 0.0;
   out_6658005824480305271[226] = 0.0;
   out_6658005824480305271[227] = 0.0;
   out_6658005824480305271[228] = 1.0;
   out_6658005824480305271[229] = 0.0;
   out_6658005824480305271[230] = 0.0;
   out_6658005824480305271[231] = 0.0;
   out_6658005824480305271[232] = 0.0;
   out_6658005824480305271[233] = 0.0;
   out_6658005824480305271[234] = 0.0;
   out_6658005824480305271[235] = 0.0;
   out_6658005824480305271[236] = 0.0;
   out_6658005824480305271[237] = 0.0;
   out_6658005824480305271[238] = 0.0;
   out_6658005824480305271[239] = 0.0;
   out_6658005824480305271[240] = 0.0;
   out_6658005824480305271[241] = 0.0;
   out_6658005824480305271[242] = 0.0;
   out_6658005824480305271[243] = 0.0;
   out_6658005824480305271[244] = 0.0;
   out_6658005824480305271[245] = 0.0;
   out_6658005824480305271[246] = 0.0;
   out_6658005824480305271[247] = 1.0;
   out_6658005824480305271[248] = 0.0;
   out_6658005824480305271[249] = 0.0;
   out_6658005824480305271[250] = 0.0;
   out_6658005824480305271[251] = 0.0;
   out_6658005824480305271[252] = 0.0;
   out_6658005824480305271[253] = 0.0;
   out_6658005824480305271[254] = 0.0;
   out_6658005824480305271[255] = 0.0;
   out_6658005824480305271[256] = 0.0;
   out_6658005824480305271[257] = 0.0;
   out_6658005824480305271[258] = 0.0;
   out_6658005824480305271[259] = 0.0;
   out_6658005824480305271[260] = 0.0;
   out_6658005824480305271[261] = 0.0;
   out_6658005824480305271[262] = 0.0;
   out_6658005824480305271[263] = 0.0;
   out_6658005824480305271[264] = 0.0;
   out_6658005824480305271[265] = 0.0;
   out_6658005824480305271[266] = 1.0;
   out_6658005824480305271[267] = 0.0;
   out_6658005824480305271[268] = 0.0;
   out_6658005824480305271[269] = 0.0;
   out_6658005824480305271[270] = 0.0;
   out_6658005824480305271[271] = 0.0;
   out_6658005824480305271[272] = 0.0;
   out_6658005824480305271[273] = 0.0;
   out_6658005824480305271[274] = 0.0;
   out_6658005824480305271[275] = 0.0;
   out_6658005824480305271[276] = 0.0;
   out_6658005824480305271[277] = 0.0;
   out_6658005824480305271[278] = 0.0;
   out_6658005824480305271[279] = 0.0;
   out_6658005824480305271[280] = 0.0;
   out_6658005824480305271[281] = 0.0;
   out_6658005824480305271[282] = 0.0;
   out_6658005824480305271[283] = 0.0;
   out_6658005824480305271[284] = 0.0;
   out_6658005824480305271[285] = 1.0;
   out_6658005824480305271[286] = 0.0;
   out_6658005824480305271[287] = 0.0;
   out_6658005824480305271[288] = 0.0;
   out_6658005824480305271[289] = 0.0;
   out_6658005824480305271[290] = 0.0;
   out_6658005824480305271[291] = 0.0;
   out_6658005824480305271[292] = 0.0;
   out_6658005824480305271[293] = 0.0;
   out_6658005824480305271[294] = 0.0;
   out_6658005824480305271[295] = 0.0;
   out_6658005824480305271[296] = 0.0;
   out_6658005824480305271[297] = 0.0;
   out_6658005824480305271[298] = 0.0;
   out_6658005824480305271[299] = 0.0;
   out_6658005824480305271[300] = 0.0;
   out_6658005824480305271[301] = 0.0;
   out_6658005824480305271[302] = 0.0;
   out_6658005824480305271[303] = 0.0;
   out_6658005824480305271[304] = 1.0;
   out_6658005824480305271[305] = 0.0;
   out_6658005824480305271[306] = 0.0;
   out_6658005824480305271[307] = 0.0;
   out_6658005824480305271[308] = 0.0;
   out_6658005824480305271[309] = 0.0;
   out_6658005824480305271[310] = 0.0;
   out_6658005824480305271[311] = 0.0;
   out_6658005824480305271[312] = 0.0;
   out_6658005824480305271[313] = 0.0;
   out_6658005824480305271[314] = 0.0;
   out_6658005824480305271[315] = 0.0;
   out_6658005824480305271[316] = 0.0;
   out_6658005824480305271[317] = 0.0;
   out_6658005824480305271[318] = 0.0;
   out_6658005824480305271[319] = 0.0;
   out_6658005824480305271[320] = 0.0;
   out_6658005824480305271[321] = 0.0;
   out_6658005824480305271[322] = 0.0;
   out_6658005824480305271[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_7993529346017171902) {
   out_7993529346017171902[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_7993529346017171902[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_7993529346017171902[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_7993529346017171902[3] = dt*state[12] + state[3];
   out_7993529346017171902[4] = dt*state[13] + state[4];
   out_7993529346017171902[5] = dt*state[14] + state[5];
   out_7993529346017171902[6] = state[6];
   out_7993529346017171902[7] = state[7];
   out_7993529346017171902[8] = state[8];
   out_7993529346017171902[9] = state[9];
   out_7993529346017171902[10] = state[10];
   out_7993529346017171902[11] = state[11];
   out_7993529346017171902[12] = state[12];
   out_7993529346017171902[13] = state[13];
   out_7993529346017171902[14] = state[14];
   out_7993529346017171902[15] = state[15];
   out_7993529346017171902[16] = state[16];
   out_7993529346017171902[17] = state[17];
}
void F_fun(double *state, double dt, double *out_348990674142262734) {
   out_348990674142262734[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_348990674142262734[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_348990674142262734[2] = 0;
   out_348990674142262734[3] = 0;
   out_348990674142262734[4] = 0;
   out_348990674142262734[5] = 0;
   out_348990674142262734[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_348990674142262734[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_348990674142262734[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_348990674142262734[9] = 0;
   out_348990674142262734[10] = 0;
   out_348990674142262734[11] = 0;
   out_348990674142262734[12] = 0;
   out_348990674142262734[13] = 0;
   out_348990674142262734[14] = 0;
   out_348990674142262734[15] = 0;
   out_348990674142262734[16] = 0;
   out_348990674142262734[17] = 0;
   out_348990674142262734[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_348990674142262734[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_348990674142262734[20] = 0;
   out_348990674142262734[21] = 0;
   out_348990674142262734[22] = 0;
   out_348990674142262734[23] = 0;
   out_348990674142262734[24] = 0;
   out_348990674142262734[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_348990674142262734[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_348990674142262734[27] = 0;
   out_348990674142262734[28] = 0;
   out_348990674142262734[29] = 0;
   out_348990674142262734[30] = 0;
   out_348990674142262734[31] = 0;
   out_348990674142262734[32] = 0;
   out_348990674142262734[33] = 0;
   out_348990674142262734[34] = 0;
   out_348990674142262734[35] = 0;
   out_348990674142262734[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_348990674142262734[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_348990674142262734[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_348990674142262734[39] = 0;
   out_348990674142262734[40] = 0;
   out_348990674142262734[41] = 0;
   out_348990674142262734[42] = 0;
   out_348990674142262734[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_348990674142262734[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_348990674142262734[45] = 0;
   out_348990674142262734[46] = 0;
   out_348990674142262734[47] = 0;
   out_348990674142262734[48] = 0;
   out_348990674142262734[49] = 0;
   out_348990674142262734[50] = 0;
   out_348990674142262734[51] = 0;
   out_348990674142262734[52] = 0;
   out_348990674142262734[53] = 0;
   out_348990674142262734[54] = 0;
   out_348990674142262734[55] = 0;
   out_348990674142262734[56] = 0;
   out_348990674142262734[57] = 1;
   out_348990674142262734[58] = 0;
   out_348990674142262734[59] = 0;
   out_348990674142262734[60] = 0;
   out_348990674142262734[61] = 0;
   out_348990674142262734[62] = 0;
   out_348990674142262734[63] = 0;
   out_348990674142262734[64] = 0;
   out_348990674142262734[65] = 0;
   out_348990674142262734[66] = dt;
   out_348990674142262734[67] = 0;
   out_348990674142262734[68] = 0;
   out_348990674142262734[69] = 0;
   out_348990674142262734[70] = 0;
   out_348990674142262734[71] = 0;
   out_348990674142262734[72] = 0;
   out_348990674142262734[73] = 0;
   out_348990674142262734[74] = 0;
   out_348990674142262734[75] = 0;
   out_348990674142262734[76] = 1;
   out_348990674142262734[77] = 0;
   out_348990674142262734[78] = 0;
   out_348990674142262734[79] = 0;
   out_348990674142262734[80] = 0;
   out_348990674142262734[81] = 0;
   out_348990674142262734[82] = 0;
   out_348990674142262734[83] = 0;
   out_348990674142262734[84] = 0;
   out_348990674142262734[85] = dt;
   out_348990674142262734[86] = 0;
   out_348990674142262734[87] = 0;
   out_348990674142262734[88] = 0;
   out_348990674142262734[89] = 0;
   out_348990674142262734[90] = 0;
   out_348990674142262734[91] = 0;
   out_348990674142262734[92] = 0;
   out_348990674142262734[93] = 0;
   out_348990674142262734[94] = 0;
   out_348990674142262734[95] = 1;
   out_348990674142262734[96] = 0;
   out_348990674142262734[97] = 0;
   out_348990674142262734[98] = 0;
   out_348990674142262734[99] = 0;
   out_348990674142262734[100] = 0;
   out_348990674142262734[101] = 0;
   out_348990674142262734[102] = 0;
   out_348990674142262734[103] = 0;
   out_348990674142262734[104] = dt;
   out_348990674142262734[105] = 0;
   out_348990674142262734[106] = 0;
   out_348990674142262734[107] = 0;
   out_348990674142262734[108] = 0;
   out_348990674142262734[109] = 0;
   out_348990674142262734[110] = 0;
   out_348990674142262734[111] = 0;
   out_348990674142262734[112] = 0;
   out_348990674142262734[113] = 0;
   out_348990674142262734[114] = 1;
   out_348990674142262734[115] = 0;
   out_348990674142262734[116] = 0;
   out_348990674142262734[117] = 0;
   out_348990674142262734[118] = 0;
   out_348990674142262734[119] = 0;
   out_348990674142262734[120] = 0;
   out_348990674142262734[121] = 0;
   out_348990674142262734[122] = 0;
   out_348990674142262734[123] = 0;
   out_348990674142262734[124] = 0;
   out_348990674142262734[125] = 0;
   out_348990674142262734[126] = 0;
   out_348990674142262734[127] = 0;
   out_348990674142262734[128] = 0;
   out_348990674142262734[129] = 0;
   out_348990674142262734[130] = 0;
   out_348990674142262734[131] = 0;
   out_348990674142262734[132] = 0;
   out_348990674142262734[133] = 1;
   out_348990674142262734[134] = 0;
   out_348990674142262734[135] = 0;
   out_348990674142262734[136] = 0;
   out_348990674142262734[137] = 0;
   out_348990674142262734[138] = 0;
   out_348990674142262734[139] = 0;
   out_348990674142262734[140] = 0;
   out_348990674142262734[141] = 0;
   out_348990674142262734[142] = 0;
   out_348990674142262734[143] = 0;
   out_348990674142262734[144] = 0;
   out_348990674142262734[145] = 0;
   out_348990674142262734[146] = 0;
   out_348990674142262734[147] = 0;
   out_348990674142262734[148] = 0;
   out_348990674142262734[149] = 0;
   out_348990674142262734[150] = 0;
   out_348990674142262734[151] = 0;
   out_348990674142262734[152] = 1;
   out_348990674142262734[153] = 0;
   out_348990674142262734[154] = 0;
   out_348990674142262734[155] = 0;
   out_348990674142262734[156] = 0;
   out_348990674142262734[157] = 0;
   out_348990674142262734[158] = 0;
   out_348990674142262734[159] = 0;
   out_348990674142262734[160] = 0;
   out_348990674142262734[161] = 0;
   out_348990674142262734[162] = 0;
   out_348990674142262734[163] = 0;
   out_348990674142262734[164] = 0;
   out_348990674142262734[165] = 0;
   out_348990674142262734[166] = 0;
   out_348990674142262734[167] = 0;
   out_348990674142262734[168] = 0;
   out_348990674142262734[169] = 0;
   out_348990674142262734[170] = 0;
   out_348990674142262734[171] = 1;
   out_348990674142262734[172] = 0;
   out_348990674142262734[173] = 0;
   out_348990674142262734[174] = 0;
   out_348990674142262734[175] = 0;
   out_348990674142262734[176] = 0;
   out_348990674142262734[177] = 0;
   out_348990674142262734[178] = 0;
   out_348990674142262734[179] = 0;
   out_348990674142262734[180] = 0;
   out_348990674142262734[181] = 0;
   out_348990674142262734[182] = 0;
   out_348990674142262734[183] = 0;
   out_348990674142262734[184] = 0;
   out_348990674142262734[185] = 0;
   out_348990674142262734[186] = 0;
   out_348990674142262734[187] = 0;
   out_348990674142262734[188] = 0;
   out_348990674142262734[189] = 0;
   out_348990674142262734[190] = 1;
   out_348990674142262734[191] = 0;
   out_348990674142262734[192] = 0;
   out_348990674142262734[193] = 0;
   out_348990674142262734[194] = 0;
   out_348990674142262734[195] = 0;
   out_348990674142262734[196] = 0;
   out_348990674142262734[197] = 0;
   out_348990674142262734[198] = 0;
   out_348990674142262734[199] = 0;
   out_348990674142262734[200] = 0;
   out_348990674142262734[201] = 0;
   out_348990674142262734[202] = 0;
   out_348990674142262734[203] = 0;
   out_348990674142262734[204] = 0;
   out_348990674142262734[205] = 0;
   out_348990674142262734[206] = 0;
   out_348990674142262734[207] = 0;
   out_348990674142262734[208] = 0;
   out_348990674142262734[209] = 1;
   out_348990674142262734[210] = 0;
   out_348990674142262734[211] = 0;
   out_348990674142262734[212] = 0;
   out_348990674142262734[213] = 0;
   out_348990674142262734[214] = 0;
   out_348990674142262734[215] = 0;
   out_348990674142262734[216] = 0;
   out_348990674142262734[217] = 0;
   out_348990674142262734[218] = 0;
   out_348990674142262734[219] = 0;
   out_348990674142262734[220] = 0;
   out_348990674142262734[221] = 0;
   out_348990674142262734[222] = 0;
   out_348990674142262734[223] = 0;
   out_348990674142262734[224] = 0;
   out_348990674142262734[225] = 0;
   out_348990674142262734[226] = 0;
   out_348990674142262734[227] = 0;
   out_348990674142262734[228] = 1;
   out_348990674142262734[229] = 0;
   out_348990674142262734[230] = 0;
   out_348990674142262734[231] = 0;
   out_348990674142262734[232] = 0;
   out_348990674142262734[233] = 0;
   out_348990674142262734[234] = 0;
   out_348990674142262734[235] = 0;
   out_348990674142262734[236] = 0;
   out_348990674142262734[237] = 0;
   out_348990674142262734[238] = 0;
   out_348990674142262734[239] = 0;
   out_348990674142262734[240] = 0;
   out_348990674142262734[241] = 0;
   out_348990674142262734[242] = 0;
   out_348990674142262734[243] = 0;
   out_348990674142262734[244] = 0;
   out_348990674142262734[245] = 0;
   out_348990674142262734[246] = 0;
   out_348990674142262734[247] = 1;
   out_348990674142262734[248] = 0;
   out_348990674142262734[249] = 0;
   out_348990674142262734[250] = 0;
   out_348990674142262734[251] = 0;
   out_348990674142262734[252] = 0;
   out_348990674142262734[253] = 0;
   out_348990674142262734[254] = 0;
   out_348990674142262734[255] = 0;
   out_348990674142262734[256] = 0;
   out_348990674142262734[257] = 0;
   out_348990674142262734[258] = 0;
   out_348990674142262734[259] = 0;
   out_348990674142262734[260] = 0;
   out_348990674142262734[261] = 0;
   out_348990674142262734[262] = 0;
   out_348990674142262734[263] = 0;
   out_348990674142262734[264] = 0;
   out_348990674142262734[265] = 0;
   out_348990674142262734[266] = 1;
   out_348990674142262734[267] = 0;
   out_348990674142262734[268] = 0;
   out_348990674142262734[269] = 0;
   out_348990674142262734[270] = 0;
   out_348990674142262734[271] = 0;
   out_348990674142262734[272] = 0;
   out_348990674142262734[273] = 0;
   out_348990674142262734[274] = 0;
   out_348990674142262734[275] = 0;
   out_348990674142262734[276] = 0;
   out_348990674142262734[277] = 0;
   out_348990674142262734[278] = 0;
   out_348990674142262734[279] = 0;
   out_348990674142262734[280] = 0;
   out_348990674142262734[281] = 0;
   out_348990674142262734[282] = 0;
   out_348990674142262734[283] = 0;
   out_348990674142262734[284] = 0;
   out_348990674142262734[285] = 1;
   out_348990674142262734[286] = 0;
   out_348990674142262734[287] = 0;
   out_348990674142262734[288] = 0;
   out_348990674142262734[289] = 0;
   out_348990674142262734[290] = 0;
   out_348990674142262734[291] = 0;
   out_348990674142262734[292] = 0;
   out_348990674142262734[293] = 0;
   out_348990674142262734[294] = 0;
   out_348990674142262734[295] = 0;
   out_348990674142262734[296] = 0;
   out_348990674142262734[297] = 0;
   out_348990674142262734[298] = 0;
   out_348990674142262734[299] = 0;
   out_348990674142262734[300] = 0;
   out_348990674142262734[301] = 0;
   out_348990674142262734[302] = 0;
   out_348990674142262734[303] = 0;
   out_348990674142262734[304] = 1;
   out_348990674142262734[305] = 0;
   out_348990674142262734[306] = 0;
   out_348990674142262734[307] = 0;
   out_348990674142262734[308] = 0;
   out_348990674142262734[309] = 0;
   out_348990674142262734[310] = 0;
   out_348990674142262734[311] = 0;
   out_348990674142262734[312] = 0;
   out_348990674142262734[313] = 0;
   out_348990674142262734[314] = 0;
   out_348990674142262734[315] = 0;
   out_348990674142262734[316] = 0;
   out_348990674142262734[317] = 0;
   out_348990674142262734[318] = 0;
   out_348990674142262734[319] = 0;
   out_348990674142262734[320] = 0;
   out_348990674142262734[321] = 0;
   out_348990674142262734[322] = 0;
   out_348990674142262734[323] = 1;
}
void h_4(double *state, double *unused, double *out_6484339586194182173) {
   out_6484339586194182173[0] = state[6] + state[9];
   out_6484339586194182173[1] = state[7] + state[10];
   out_6484339586194182173[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_2286355841577516071) {
   out_2286355841577516071[0] = 0;
   out_2286355841577516071[1] = 0;
   out_2286355841577516071[2] = 0;
   out_2286355841577516071[3] = 0;
   out_2286355841577516071[4] = 0;
   out_2286355841577516071[5] = 0;
   out_2286355841577516071[6] = 1;
   out_2286355841577516071[7] = 0;
   out_2286355841577516071[8] = 0;
   out_2286355841577516071[9] = 1;
   out_2286355841577516071[10] = 0;
   out_2286355841577516071[11] = 0;
   out_2286355841577516071[12] = 0;
   out_2286355841577516071[13] = 0;
   out_2286355841577516071[14] = 0;
   out_2286355841577516071[15] = 0;
   out_2286355841577516071[16] = 0;
   out_2286355841577516071[17] = 0;
   out_2286355841577516071[18] = 0;
   out_2286355841577516071[19] = 0;
   out_2286355841577516071[20] = 0;
   out_2286355841577516071[21] = 0;
   out_2286355841577516071[22] = 0;
   out_2286355841577516071[23] = 0;
   out_2286355841577516071[24] = 0;
   out_2286355841577516071[25] = 1;
   out_2286355841577516071[26] = 0;
   out_2286355841577516071[27] = 0;
   out_2286355841577516071[28] = 1;
   out_2286355841577516071[29] = 0;
   out_2286355841577516071[30] = 0;
   out_2286355841577516071[31] = 0;
   out_2286355841577516071[32] = 0;
   out_2286355841577516071[33] = 0;
   out_2286355841577516071[34] = 0;
   out_2286355841577516071[35] = 0;
   out_2286355841577516071[36] = 0;
   out_2286355841577516071[37] = 0;
   out_2286355841577516071[38] = 0;
   out_2286355841577516071[39] = 0;
   out_2286355841577516071[40] = 0;
   out_2286355841577516071[41] = 0;
   out_2286355841577516071[42] = 0;
   out_2286355841577516071[43] = 0;
   out_2286355841577516071[44] = 1;
   out_2286355841577516071[45] = 0;
   out_2286355841577516071[46] = 0;
   out_2286355841577516071[47] = 1;
   out_2286355841577516071[48] = 0;
   out_2286355841577516071[49] = 0;
   out_2286355841577516071[50] = 0;
   out_2286355841577516071[51] = 0;
   out_2286355841577516071[52] = 0;
   out_2286355841577516071[53] = 0;
}
void h_10(double *state, double *unused, double *out_2411329706835363596) {
   out_2411329706835363596[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_2411329706835363596[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_2411329706835363596[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4215648645313872337) {
   out_4215648645313872337[0] = 0;
   out_4215648645313872337[1] = 9.8100000000000005*cos(state[1]);
   out_4215648645313872337[2] = 0;
   out_4215648645313872337[3] = 0;
   out_4215648645313872337[4] = -state[8];
   out_4215648645313872337[5] = state[7];
   out_4215648645313872337[6] = 0;
   out_4215648645313872337[7] = state[5];
   out_4215648645313872337[8] = -state[4];
   out_4215648645313872337[9] = 0;
   out_4215648645313872337[10] = 0;
   out_4215648645313872337[11] = 0;
   out_4215648645313872337[12] = 1;
   out_4215648645313872337[13] = 0;
   out_4215648645313872337[14] = 0;
   out_4215648645313872337[15] = 1;
   out_4215648645313872337[16] = 0;
   out_4215648645313872337[17] = 0;
   out_4215648645313872337[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4215648645313872337[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4215648645313872337[20] = 0;
   out_4215648645313872337[21] = state[8];
   out_4215648645313872337[22] = 0;
   out_4215648645313872337[23] = -state[6];
   out_4215648645313872337[24] = -state[5];
   out_4215648645313872337[25] = 0;
   out_4215648645313872337[26] = state[3];
   out_4215648645313872337[27] = 0;
   out_4215648645313872337[28] = 0;
   out_4215648645313872337[29] = 0;
   out_4215648645313872337[30] = 0;
   out_4215648645313872337[31] = 1;
   out_4215648645313872337[32] = 0;
   out_4215648645313872337[33] = 0;
   out_4215648645313872337[34] = 1;
   out_4215648645313872337[35] = 0;
   out_4215648645313872337[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4215648645313872337[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4215648645313872337[38] = 0;
   out_4215648645313872337[39] = -state[7];
   out_4215648645313872337[40] = state[6];
   out_4215648645313872337[41] = 0;
   out_4215648645313872337[42] = state[4];
   out_4215648645313872337[43] = -state[3];
   out_4215648645313872337[44] = 0;
   out_4215648645313872337[45] = 0;
   out_4215648645313872337[46] = 0;
   out_4215648645313872337[47] = 0;
   out_4215648645313872337[48] = 0;
   out_4215648645313872337[49] = 0;
   out_4215648645313872337[50] = 1;
   out_4215648645313872337[51] = 0;
   out_4215648645313872337[52] = 0;
   out_4215648645313872337[53] = 1;
}
void h_13(double *state, double *unused, double *out_7156475162968172826) {
   out_7156475162968172826[0] = state[3];
   out_7156475162968172826[1] = state[4];
   out_7156475162968172826[2] = state[5];
}
void H_13(double *state, double *unused, double *out_925917983754816730) {
   out_925917983754816730[0] = 0;
   out_925917983754816730[1] = 0;
   out_925917983754816730[2] = 0;
   out_925917983754816730[3] = 1;
   out_925917983754816730[4] = 0;
   out_925917983754816730[5] = 0;
   out_925917983754816730[6] = 0;
   out_925917983754816730[7] = 0;
   out_925917983754816730[8] = 0;
   out_925917983754816730[9] = 0;
   out_925917983754816730[10] = 0;
   out_925917983754816730[11] = 0;
   out_925917983754816730[12] = 0;
   out_925917983754816730[13] = 0;
   out_925917983754816730[14] = 0;
   out_925917983754816730[15] = 0;
   out_925917983754816730[16] = 0;
   out_925917983754816730[17] = 0;
   out_925917983754816730[18] = 0;
   out_925917983754816730[19] = 0;
   out_925917983754816730[20] = 0;
   out_925917983754816730[21] = 0;
   out_925917983754816730[22] = 1;
   out_925917983754816730[23] = 0;
   out_925917983754816730[24] = 0;
   out_925917983754816730[25] = 0;
   out_925917983754816730[26] = 0;
   out_925917983754816730[27] = 0;
   out_925917983754816730[28] = 0;
   out_925917983754816730[29] = 0;
   out_925917983754816730[30] = 0;
   out_925917983754816730[31] = 0;
   out_925917983754816730[32] = 0;
   out_925917983754816730[33] = 0;
   out_925917983754816730[34] = 0;
   out_925917983754816730[35] = 0;
   out_925917983754816730[36] = 0;
   out_925917983754816730[37] = 0;
   out_925917983754816730[38] = 0;
   out_925917983754816730[39] = 0;
   out_925917983754816730[40] = 0;
   out_925917983754816730[41] = 1;
   out_925917983754816730[42] = 0;
   out_925917983754816730[43] = 0;
   out_925917983754816730[44] = 0;
   out_925917983754816730[45] = 0;
   out_925917983754816730[46] = 0;
   out_925917983754816730[47] = 0;
   out_925917983754816730[48] = 0;
   out_925917983754816730[49] = 0;
   out_925917983754816730[50] = 0;
   out_925917983754816730[51] = 0;
   out_925917983754816730[52] = 0;
   out_925917983754816730[53] = 0;
}
void h_14(double *state, double *unused, double *out_7515135358449386116) {
   out_7515135358449386116[0] = state[6];
   out_7515135358449386116[1] = state[7];
   out_7515135358449386116[2] = state[8];
}
void H_14(double *state, double *unused, double *out_5369144273872888367) {
   out_5369144273872888367[0] = 0;
   out_5369144273872888367[1] = 0;
   out_5369144273872888367[2] = 0;
   out_5369144273872888367[3] = 0;
   out_5369144273872888367[4] = 0;
   out_5369144273872888367[5] = 0;
   out_5369144273872888367[6] = 1;
   out_5369144273872888367[7] = 0;
   out_5369144273872888367[8] = 0;
   out_5369144273872888367[9] = 0;
   out_5369144273872888367[10] = 0;
   out_5369144273872888367[11] = 0;
   out_5369144273872888367[12] = 0;
   out_5369144273872888367[13] = 0;
   out_5369144273872888367[14] = 0;
   out_5369144273872888367[15] = 0;
   out_5369144273872888367[16] = 0;
   out_5369144273872888367[17] = 0;
   out_5369144273872888367[18] = 0;
   out_5369144273872888367[19] = 0;
   out_5369144273872888367[20] = 0;
   out_5369144273872888367[21] = 0;
   out_5369144273872888367[22] = 0;
   out_5369144273872888367[23] = 0;
   out_5369144273872888367[24] = 0;
   out_5369144273872888367[25] = 1;
   out_5369144273872888367[26] = 0;
   out_5369144273872888367[27] = 0;
   out_5369144273872888367[28] = 0;
   out_5369144273872888367[29] = 0;
   out_5369144273872888367[30] = 0;
   out_5369144273872888367[31] = 0;
   out_5369144273872888367[32] = 0;
   out_5369144273872888367[33] = 0;
   out_5369144273872888367[34] = 0;
   out_5369144273872888367[35] = 0;
   out_5369144273872888367[36] = 0;
   out_5369144273872888367[37] = 0;
   out_5369144273872888367[38] = 0;
   out_5369144273872888367[39] = 0;
   out_5369144273872888367[40] = 0;
   out_5369144273872888367[41] = 0;
   out_5369144273872888367[42] = 0;
   out_5369144273872888367[43] = 0;
   out_5369144273872888367[44] = 1;
   out_5369144273872888367[45] = 0;
   out_5369144273872888367[46] = 0;
   out_5369144273872888367[47] = 0;
   out_5369144273872888367[48] = 0;
   out_5369144273872888367[49] = 0;
   out_5369144273872888367[50] = 0;
   out_5369144273872888367[51] = 0;
   out_5369144273872888367[52] = 0;
   out_5369144273872888367[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_4215104848141748955) {
  err_fun(nom_x, delta_x, out_4215104848141748955);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_8045931564468893177) {
  inv_err_fun(nom_x, true_x, out_8045931564468893177);
}
void pose_H_mod_fun(double *state, double *out_6658005824480305271) {
  H_mod_fun(state, out_6658005824480305271);
}
void pose_f_fun(double *state, double dt, double *out_7993529346017171902) {
  f_fun(state,  dt, out_7993529346017171902);
}
void pose_F_fun(double *state, double dt, double *out_348990674142262734) {
  F_fun(state,  dt, out_348990674142262734);
}
void pose_h_4(double *state, double *unused, double *out_6484339586194182173) {
  h_4(state, unused, out_6484339586194182173);
}
void pose_H_4(double *state, double *unused, double *out_2286355841577516071) {
  H_4(state, unused, out_2286355841577516071);
}
void pose_h_10(double *state, double *unused, double *out_2411329706835363596) {
  h_10(state, unused, out_2411329706835363596);
}
void pose_H_10(double *state, double *unused, double *out_4215648645313872337) {
  H_10(state, unused, out_4215648645313872337);
}
void pose_h_13(double *state, double *unused, double *out_7156475162968172826) {
  h_13(state, unused, out_7156475162968172826);
}
void pose_H_13(double *state, double *unused, double *out_925917983754816730) {
  H_13(state, unused, out_925917983754816730);
}
void pose_h_14(double *state, double *unused, double *out_7515135358449386116) {
  h_14(state, unused, out_7515135358449386116);
}
void pose_H_14(double *state, double *unused, double *out_5369144273872888367) {
  H_14(state, unused, out_5369144273872888367);
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
