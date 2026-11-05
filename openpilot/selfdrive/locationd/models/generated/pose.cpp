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
void err_fun(double *nom_x, double *delta_x, double *out_5632398071682842920) {
   out_5632398071682842920[0] = delta_x[0] + nom_x[0];
   out_5632398071682842920[1] = delta_x[1] + nom_x[1];
   out_5632398071682842920[2] = delta_x[2] + nom_x[2];
   out_5632398071682842920[3] = delta_x[3] + nom_x[3];
   out_5632398071682842920[4] = delta_x[4] + nom_x[4];
   out_5632398071682842920[5] = delta_x[5] + nom_x[5];
   out_5632398071682842920[6] = delta_x[6] + nom_x[6];
   out_5632398071682842920[7] = delta_x[7] + nom_x[7];
   out_5632398071682842920[8] = delta_x[8] + nom_x[8];
   out_5632398071682842920[9] = delta_x[9] + nom_x[9];
   out_5632398071682842920[10] = delta_x[10] + nom_x[10];
   out_5632398071682842920[11] = delta_x[11] + nom_x[11];
   out_5632398071682842920[12] = delta_x[12] + nom_x[12];
   out_5632398071682842920[13] = delta_x[13] + nom_x[13];
   out_5632398071682842920[14] = delta_x[14] + nom_x[14];
   out_5632398071682842920[15] = delta_x[15] + nom_x[15];
   out_5632398071682842920[16] = delta_x[16] + nom_x[16];
   out_5632398071682842920[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_4793870659787952118) {
   out_4793870659787952118[0] = -nom_x[0] + true_x[0];
   out_4793870659787952118[1] = -nom_x[1] + true_x[1];
   out_4793870659787952118[2] = -nom_x[2] + true_x[2];
   out_4793870659787952118[3] = -nom_x[3] + true_x[3];
   out_4793870659787952118[4] = -nom_x[4] + true_x[4];
   out_4793870659787952118[5] = -nom_x[5] + true_x[5];
   out_4793870659787952118[6] = -nom_x[6] + true_x[6];
   out_4793870659787952118[7] = -nom_x[7] + true_x[7];
   out_4793870659787952118[8] = -nom_x[8] + true_x[8];
   out_4793870659787952118[9] = -nom_x[9] + true_x[9];
   out_4793870659787952118[10] = -nom_x[10] + true_x[10];
   out_4793870659787952118[11] = -nom_x[11] + true_x[11];
   out_4793870659787952118[12] = -nom_x[12] + true_x[12];
   out_4793870659787952118[13] = -nom_x[13] + true_x[13];
   out_4793870659787952118[14] = -nom_x[14] + true_x[14];
   out_4793870659787952118[15] = -nom_x[15] + true_x[15];
   out_4793870659787952118[16] = -nom_x[16] + true_x[16];
   out_4793870659787952118[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_4028297677886627584) {
   out_4028297677886627584[0] = 1.0;
   out_4028297677886627584[1] = 0.0;
   out_4028297677886627584[2] = 0.0;
   out_4028297677886627584[3] = 0.0;
   out_4028297677886627584[4] = 0.0;
   out_4028297677886627584[5] = 0.0;
   out_4028297677886627584[6] = 0.0;
   out_4028297677886627584[7] = 0.0;
   out_4028297677886627584[8] = 0.0;
   out_4028297677886627584[9] = 0.0;
   out_4028297677886627584[10] = 0.0;
   out_4028297677886627584[11] = 0.0;
   out_4028297677886627584[12] = 0.0;
   out_4028297677886627584[13] = 0.0;
   out_4028297677886627584[14] = 0.0;
   out_4028297677886627584[15] = 0.0;
   out_4028297677886627584[16] = 0.0;
   out_4028297677886627584[17] = 0.0;
   out_4028297677886627584[18] = 0.0;
   out_4028297677886627584[19] = 1.0;
   out_4028297677886627584[20] = 0.0;
   out_4028297677886627584[21] = 0.0;
   out_4028297677886627584[22] = 0.0;
   out_4028297677886627584[23] = 0.0;
   out_4028297677886627584[24] = 0.0;
   out_4028297677886627584[25] = 0.0;
   out_4028297677886627584[26] = 0.0;
   out_4028297677886627584[27] = 0.0;
   out_4028297677886627584[28] = 0.0;
   out_4028297677886627584[29] = 0.0;
   out_4028297677886627584[30] = 0.0;
   out_4028297677886627584[31] = 0.0;
   out_4028297677886627584[32] = 0.0;
   out_4028297677886627584[33] = 0.0;
   out_4028297677886627584[34] = 0.0;
   out_4028297677886627584[35] = 0.0;
   out_4028297677886627584[36] = 0.0;
   out_4028297677886627584[37] = 0.0;
   out_4028297677886627584[38] = 1.0;
   out_4028297677886627584[39] = 0.0;
   out_4028297677886627584[40] = 0.0;
   out_4028297677886627584[41] = 0.0;
   out_4028297677886627584[42] = 0.0;
   out_4028297677886627584[43] = 0.0;
   out_4028297677886627584[44] = 0.0;
   out_4028297677886627584[45] = 0.0;
   out_4028297677886627584[46] = 0.0;
   out_4028297677886627584[47] = 0.0;
   out_4028297677886627584[48] = 0.0;
   out_4028297677886627584[49] = 0.0;
   out_4028297677886627584[50] = 0.0;
   out_4028297677886627584[51] = 0.0;
   out_4028297677886627584[52] = 0.0;
   out_4028297677886627584[53] = 0.0;
   out_4028297677886627584[54] = 0.0;
   out_4028297677886627584[55] = 0.0;
   out_4028297677886627584[56] = 0.0;
   out_4028297677886627584[57] = 1.0;
   out_4028297677886627584[58] = 0.0;
   out_4028297677886627584[59] = 0.0;
   out_4028297677886627584[60] = 0.0;
   out_4028297677886627584[61] = 0.0;
   out_4028297677886627584[62] = 0.0;
   out_4028297677886627584[63] = 0.0;
   out_4028297677886627584[64] = 0.0;
   out_4028297677886627584[65] = 0.0;
   out_4028297677886627584[66] = 0.0;
   out_4028297677886627584[67] = 0.0;
   out_4028297677886627584[68] = 0.0;
   out_4028297677886627584[69] = 0.0;
   out_4028297677886627584[70] = 0.0;
   out_4028297677886627584[71] = 0.0;
   out_4028297677886627584[72] = 0.0;
   out_4028297677886627584[73] = 0.0;
   out_4028297677886627584[74] = 0.0;
   out_4028297677886627584[75] = 0.0;
   out_4028297677886627584[76] = 1.0;
   out_4028297677886627584[77] = 0.0;
   out_4028297677886627584[78] = 0.0;
   out_4028297677886627584[79] = 0.0;
   out_4028297677886627584[80] = 0.0;
   out_4028297677886627584[81] = 0.0;
   out_4028297677886627584[82] = 0.0;
   out_4028297677886627584[83] = 0.0;
   out_4028297677886627584[84] = 0.0;
   out_4028297677886627584[85] = 0.0;
   out_4028297677886627584[86] = 0.0;
   out_4028297677886627584[87] = 0.0;
   out_4028297677886627584[88] = 0.0;
   out_4028297677886627584[89] = 0.0;
   out_4028297677886627584[90] = 0.0;
   out_4028297677886627584[91] = 0.0;
   out_4028297677886627584[92] = 0.0;
   out_4028297677886627584[93] = 0.0;
   out_4028297677886627584[94] = 0.0;
   out_4028297677886627584[95] = 1.0;
   out_4028297677886627584[96] = 0.0;
   out_4028297677886627584[97] = 0.0;
   out_4028297677886627584[98] = 0.0;
   out_4028297677886627584[99] = 0.0;
   out_4028297677886627584[100] = 0.0;
   out_4028297677886627584[101] = 0.0;
   out_4028297677886627584[102] = 0.0;
   out_4028297677886627584[103] = 0.0;
   out_4028297677886627584[104] = 0.0;
   out_4028297677886627584[105] = 0.0;
   out_4028297677886627584[106] = 0.0;
   out_4028297677886627584[107] = 0.0;
   out_4028297677886627584[108] = 0.0;
   out_4028297677886627584[109] = 0.0;
   out_4028297677886627584[110] = 0.0;
   out_4028297677886627584[111] = 0.0;
   out_4028297677886627584[112] = 0.0;
   out_4028297677886627584[113] = 0.0;
   out_4028297677886627584[114] = 1.0;
   out_4028297677886627584[115] = 0.0;
   out_4028297677886627584[116] = 0.0;
   out_4028297677886627584[117] = 0.0;
   out_4028297677886627584[118] = 0.0;
   out_4028297677886627584[119] = 0.0;
   out_4028297677886627584[120] = 0.0;
   out_4028297677886627584[121] = 0.0;
   out_4028297677886627584[122] = 0.0;
   out_4028297677886627584[123] = 0.0;
   out_4028297677886627584[124] = 0.0;
   out_4028297677886627584[125] = 0.0;
   out_4028297677886627584[126] = 0.0;
   out_4028297677886627584[127] = 0.0;
   out_4028297677886627584[128] = 0.0;
   out_4028297677886627584[129] = 0.0;
   out_4028297677886627584[130] = 0.0;
   out_4028297677886627584[131] = 0.0;
   out_4028297677886627584[132] = 0.0;
   out_4028297677886627584[133] = 1.0;
   out_4028297677886627584[134] = 0.0;
   out_4028297677886627584[135] = 0.0;
   out_4028297677886627584[136] = 0.0;
   out_4028297677886627584[137] = 0.0;
   out_4028297677886627584[138] = 0.0;
   out_4028297677886627584[139] = 0.0;
   out_4028297677886627584[140] = 0.0;
   out_4028297677886627584[141] = 0.0;
   out_4028297677886627584[142] = 0.0;
   out_4028297677886627584[143] = 0.0;
   out_4028297677886627584[144] = 0.0;
   out_4028297677886627584[145] = 0.0;
   out_4028297677886627584[146] = 0.0;
   out_4028297677886627584[147] = 0.0;
   out_4028297677886627584[148] = 0.0;
   out_4028297677886627584[149] = 0.0;
   out_4028297677886627584[150] = 0.0;
   out_4028297677886627584[151] = 0.0;
   out_4028297677886627584[152] = 1.0;
   out_4028297677886627584[153] = 0.0;
   out_4028297677886627584[154] = 0.0;
   out_4028297677886627584[155] = 0.0;
   out_4028297677886627584[156] = 0.0;
   out_4028297677886627584[157] = 0.0;
   out_4028297677886627584[158] = 0.0;
   out_4028297677886627584[159] = 0.0;
   out_4028297677886627584[160] = 0.0;
   out_4028297677886627584[161] = 0.0;
   out_4028297677886627584[162] = 0.0;
   out_4028297677886627584[163] = 0.0;
   out_4028297677886627584[164] = 0.0;
   out_4028297677886627584[165] = 0.0;
   out_4028297677886627584[166] = 0.0;
   out_4028297677886627584[167] = 0.0;
   out_4028297677886627584[168] = 0.0;
   out_4028297677886627584[169] = 0.0;
   out_4028297677886627584[170] = 0.0;
   out_4028297677886627584[171] = 1.0;
   out_4028297677886627584[172] = 0.0;
   out_4028297677886627584[173] = 0.0;
   out_4028297677886627584[174] = 0.0;
   out_4028297677886627584[175] = 0.0;
   out_4028297677886627584[176] = 0.0;
   out_4028297677886627584[177] = 0.0;
   out_4028297677886627584[178] = 0.0;
   out_4028297677886627584[179] = 0.0;
   out_4028297677886627584[180] = 0.0;
   out_4028297677886627584[181] = 0.0;
   out_4028297677886627584[182] = 0.0;
   out_4028297677886627584[183] = 0.0;
   out_4028297677886627584[184] = 0.0;
   out_4028297677886627584[185] = 0.0;
   out_4028297677886627584[186] = 0.0;
   out_4028297677886627584[187] = 0.0;
   out_4028297677886627584[188] = 0.0;
   out_4028297677886627584[189] = 0.0;
   out_4028297677886627584[190] = 1.0;
   out_4028297677886627584[191] = 0.0;
   out_4028297677886627584[192] = 0.0;
   out_4028297677886627584[193] = 0.0;
   out_4028297677886627584[194] = 0.0;
   out_4028297677886627584[195] = 0.0;
   out_4028297677886627584[196] = 0.0;
   out_4028297677886627584[197] = 0.0;
   out_4028297677886627584[198] = 0.0;
   out_4028297677886627584[199] = 0.0;
   out_4028297677886627584[200] = 0.0;
   out_4028297677886627584[201] = 0.0;
   out_4028297677886627584[202] = 0.0;
   out_4028297677886627584[203] = 0.0;
   out_4028297677886627584[204] = 0.0;
   out_4028297677886627584[205] = 0.0;
   out_4028297677886627584[206] = 0.0;
   out_4028297677886627584[207] = 0.0;
   out_4028297677886627584[208] = 0.0;
   out_4028297677886627584[209] = 1.0;
   out_4028297677886627584[210] = 0.0;
   out_4028297677886627584[211] = 0.0;
   out_4028297677886627584[212] = 0.0;
   out_4028297677886627584[213] = 0.0;
   out_4028297677886627584[214] = 0.0;
   out_4028297677886627584[215] = 0.0;
   out_4028297677886627584[216] = 0.0;
   out_4028297677886627584[217] = 0.0;
   out_4028297677886627584[218] = 0.0;
   out_4028297677886627584[219] = 0.0;
   out_4028297677886627584[220] = 0.0;
   out_4028297677886627584[221] = 0.0;
   out_4028297677886627584[222] = 0.0;
   out_4028297677886627584[223] = 0.0;
   out_4028297677886627584[224] = 0.0;
   out_4028297677886627584[225] = 0.0;
   out_4028297677886627584[226] = 0.0;
   out_4028297677886627584[227] = 0.0;
   out_4028297677886627584[228] = 1.0;
   out_4028297677886627584[229] = 0.0;
   out_4028297677886627584[230] = 0.0;
   out_4028297677886627584[231] = 0.0;
   out_4028297677886627584[232] = 0.0;
   out_4028297677886627584[233] = 0.0;
   out_4028297677886627584[234] = 0.0;
   out_4028297677886627584[235] = 0.0;
   out_4028297677886627584[236] = 0.0;
   out_4028297677886627584[237] = 0.0;
   out_4028297677886627584[238] = 0.0;
   out_4028297677886627584[239] = 0.0;
   out_4028297677886627584[240] = 0.0;
   out_4028297677886627584[241] = 0.0;
   out_4028297677886627584[242] = 0.0;
   out_4028297677886627584[243] = 0.0;
   out_4028297677886627584[244] = 0.0;
   out_4028297677886627584[245] = 0.0;
   out_4028297677886627584[246] = 0.0;
   out_4028297677886627584[247] = 1.0;
   out_4028297677886627584[248] = 0.0;
   out_4028297677886627584[249] = 0.0;
   out_4028297677886627584[250] = 0.0;
   out_4028297677886627584[251] = 0.0;
   out_4028297677886627584[252] = 0.0;
   out_4028297677886627584[253] = 0.0;
   out_4028297677886627584[254] = 0.0;
   out_4028297677886627584[255] = 0.0;
   out_4028297677886627584[256] = 0.0;
   out_4028297677886627584[257] = 0.0;
   out_4028297677886627584[258] = 0.0;
   out_4028297677886627584[259] = 0.0;
   out_4028297677886627584[260] = 0.0;
   out_4028297677886627584[261] = 0.0;
   out_4028297677886627584[262] = 0.0;
   out_4028297677886627584[263] = 0.0;
   out_4028297677886627584[264] = 0.0;
   out_4028297677886627584[265] = 0.0;
   out_4028297677886627584[266] = 1.0;
   out_4028297677886627584[267] = 0.0;
   out_4028297677886627584[268] = 0.0;
   out_4028297677886627584[269] = 0.0;
   out_4028297677886627584[270] = 0.0;
   out_4028297677886627584[271] = 0.0;
   out_4028297677886627584[272] = 0.0;
   out_4028297677886627584[273] = 0.0;
   out_4028297677886627584[274] = 0.0;
   out_4028297677886627584[275] = 0.0;
   out_4028297677886627584[276] = 0.0;
   out_4028297677886627584[277] = 0.0;
   out_4028297677886627584[278] = 0.0;
   out_4028297677886627584[279] = 0.0;
   out_4028297677886627584[280] = 0.0;
   out_4028297677886627584[281] = 0.0;
   out_4028297677886627584[282] = 0.0;
   out_4028297677886627584[283] = 0.0;
   out_4028297677886627584[284] = 0.0;
   out_4028297677886627584[285] = 1.0;
   out_4028297677886627584[286] = 0.0;
   out_4028297677886627584[287] = 0.0;
   out_4028297677886627584[288] = 0.0;
   out_4028297677886627584[289] = 0.0;
   out_4028297677886627584[290] = 0.0;
   out_4028297677886627584[291] = 0.0;
   out_4028297677886627584[292] = 0.0;
   out_4028297677886627584[293] = 0.0;
   out_4028297677886627584[294] = 0.0;
   out_4028297677886627584[295] = 0.0;
   out_4028297677886627584[296] = 0.0;
   out_4028297677886627584[297] = 0.0;
   out_4028297677886627584[298] = 0.0;
   out_4028297677886627584[299] = 0.0;
   out_4028297677886627584[300] = 0.0;
   out_4028297677886627584[301] = 0.0;
   out_4028297677886627584[302] = 0.0;
   out_4028297677886627584[303] = 0.0;
   out_4028297677886627584[304] = 1.0;
   out_4028297677886627584[305] = 0.0;
   out_4028297677886627584[306] = 0.0;
   out_4028297677886627584[307] = 0.0;
   out_4028297677886627584[308] = 0.0;
   out_4028297677886627584[309] = 0.0;
   out_4028297677886627584[310] = 0.0;
   out_4028297677886627584[311] = 0.0;
   out_4028297677886627584[312] = 0.0;
   out_4028297677886627584[313] = 0.0;
   out_4028297677886627584[314] = 0.0;
   out_4028297677886627584[315] = 0.0;
   out_4028297677886627584[316] = 0.0;
   out_4028297677886627584[317] = 0.0;
   out_4028297677886627584[318] = 0.0;
   out_4028297677886627584[319] = 0.0;
   out_4028297677886627584[320] = 0.0;
   out_4028297677886627584[321] = 0.0;
   out_4028297677886627584[322] = 0.0;
   out_4028297677886627584[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_2980855877602976982) {
   out_2980855877602976982[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_2980855877602976982[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_2980855877602976982[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_2980855877602976982[3] = dt*state[12] + state[3];
   out_2980855877602976982[4] = dt*state[13] + state[4];
   out_2980855877602976982[5] = dt*state[14] + state[5];
   out_2980855877602976982[6] = state[6];
   out_2980855877602976982[7] = state[7];
   out_2980855877602976982[8] = state[8];
   out_2980855877602976982[9] = state[9];
   out_2980855877602976982[10] = state[10];
   out_2980855877602976982[11] = state[11];
   out_2980855877602976982[12] = state[12];
   out_2980855877602976982[13] = state[13];
   out_2980855877602976982[14] = state[14];
   out_2980855877602976982[15] = state[15];
   out_2980855877602976982[16] = state[16];
   out_2980855877602976982[17] = state[17];
}
void F_fun(double *state, double dt, double *out_2442030683835704114) {
   out_2442030683835704114[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2442030683835704114[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2442030683835704114[2] = 0;
   out_2442030683835704114[3] = 0;
   out_2442030683835704114[4] = 0;
   out_2442030683835704114[5] = 0;
   out_2442030683835704114[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2442030683835704114[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2442030683835704114[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_2442030683835704114[9] = 0;
   out_2442030683835704114[10] = 0;
   out_2442030683835704114[11] = 0;
   out_2442030683835704114[12] = 0;
   out_2442030683835704114[13] = 0;
   out_2442030683835704114[14] = 0;
   out_2442030683835704114[15] = 0;
   out_2442030683835704114[16] = 0;
   out_2442030683835704114[17] = 0;
   out_2442030683835704114[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2442030683835704114[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2442030683835704114[20] = 0;
   out_2442030683835704114[21] = 0;
   out_2442030683835704114[22] = 0;
   out_2442030683835704114[23] = 0;
   out_2442030683835704114[24] = 0;
   out_2442030683835704114[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2442030683835704114[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_2442030683835704114[27] = 0;
   out_2442030683835704114[28] = 0;
   out_2442030683835704114[29] = 0;
   out_2442030683835704114[30] = 0;
   out_2442030683835704114[31] = 0;
   out_2442030683835704114[32] = 0;
   out_2442030683835704114[33] = 0;
   out_2442030683835704114[34] = 0;
   out_2442030683835704114[35] = 0;
   out_2442030683835704114[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2442030683835704114[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2442030683835704114[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2442030683835704114[39] = 0;
   out_2442030683835704114[40] = 0;
   out_2442030683835704114[41] = 0;
   out_2442030683835704114[42] = 0;
   out_2442030683835704114[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2442030683835704114[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_2442030683835704114[45] = 0;
   out_2442030683835704114[46] = 0;
   out_2442030683835704114[47] = 0;
   out_2442030683835704114[48] = 0;
   out_2442030683835704114[49] = 0;
   out_2442030683835704114[50] = 0;
   out_2442030683835704114[51] = 0;
   out_2442030683835704114[52] = 0;
   out_2442030683835704114[53] = 0;
   out_2442030683835704114[54] = 0;
   out_2442030683835704114[55] = 0;
   out_2442030683835704114[56] = 0;
   out_2442030683835704114[57] = 1;
   out_2442030683835704114[58] = 0;
   out_2442030683835704114[59] = 0;
   out_2442030683835704114[60] = 0;
   out_2442030683835704114[61] = 0;
   out_2442030683835704114[62] = 0;
   out_2442030683835704114[63] = 0;
   out_2442030683835704114[64] = 0;
   out_2442030683835704114[65] = 0;
   out_2442030683835704114[66] = dt;
   out_2442030683835704114[67] = 0;
   out_2442030683835704114[68] = 0;
   out_2442030683835704114[69] = 0;
   out_2442030683835704114[70] = 0;
   out_2442030683835704114[71] = 0;
   out_2442030683835704114[72] = 0;
   out_2442030683835704114[73] = 0;
   out_2442030683835704114[74] = 0;
   out_2442030683835704114[75] = 0;
   out_2442030683835704114[76] = 1;
   out_2442030683835704114[77] = 0;
   out_2442030683835704114[78] = 0;
   out_2442030683835704114[79] = 0;
   out_2442030683835704114[80] = 0;
   out_2442030683835704114[81] = 0;
   out_2442030683835704114[82] = 0;
   out_2442030683835704114[83] = 0;
   out_2442030683835704114[84] = 0;
   out_2442030683835704114[85] = dt;
   out_2442030683835704114[86] = 0;
   out_2442030683835704114[87] = 0;
   out_2442030683835704114[88] = 0;
   out_2442030683835704114[89] = 0;
   out_2442030683835704114[90] = 0;
   out_2442030683835704114[91] = 0;
   out_2442030683835704114[92] = 0;
   out_2442030683835704114[93] = 0;
   out_2442030683835704114[94] = 0;
   out_2442030683835704114[95] = 1;
   out_2442030683835704114[96] = 0;
   out_2442030683835704114[97] = 0;
   out_2442030683835704114[98] = 0;
   out_2442030683835704114[99] = 0;
   out_2442030683835704114[100] = 0;
   out_2442030683835704114[101] = 0;
   out_2442030683835704114[102] = 0;
   out_2442030683835704114[103] = 0;
   out_2442030683835704114[104] = dt;
   out_2442030683835704114[105] = 0;
   out_2442030683835704114[106] = 0;
   out_2442030683835704114[107] = 0;
   out_2442030683835704114[108] = 0;
   out_2442030683835704114[109] = 0;
   out_2442030683835704114[110] = 0;
   out_2442030683835704114[111] = 0;
   out_2442030683835704114[112] = 0;
   out_2442030683835704114[113] = 0;
   out_2442030683835704114[114] = 1;
   out_2442030683835704114[115] = 0;
   out_2442030683835704114[116] = 0;
   out_2442030683835704114[117] = 0;
   out_2442030683835704114[118] = 0;
   out_2442030683835704114[119] = 0;
   out_2442030683835704114[120] = 0;
   out_2442030683835704114[121] = 0;
   out_2442030683835704114[122] = 0;
   out_2442030683835704114[123] = 0;
   out_2442030683835704114[124] = 0;
   out_2442030683835704114[125] = 0;
   out_2442030683835704114[126] = 0;
   out_2442030683835704114[127] = 0;
   out_2442030683835704114[128] = 0;
   out_2442030683835704114[129] = 0;
   out_2442030683835704114[130] = 0;
   out_2442030683835704114[131] = 0;
   out_2442030683835704114[132] = 0;
   out_2442030683835704114[133] = 1;
   out_2442030683835704114[134] = 0;
   out_2442030683835704114[135] = 0;
   out_2442030683835704114[136] = 0;
   out_2442030683835704114[137] = 0;
   out_2442030683835704114[138] = 0;
   out_2442030683835704114[139] = 0;
   out_2442030683835704114[140] = 0;
   out_2442030683835704114[141] = 0;
   out_2442030683835704114[142] = 0;
   out_2442030683835704114[143] = 0;
   out_2442030683835704114[144] = 0;
   out_2442030683835704114[145] = 0;
   out_2442030683835704114[146] = 0;
   out_2442030683835704114[147] = 0;
   out_2442030683835704114[148] = 0;
   out_2442030683835704114[149] = 0;
   out_2442030683835704114[150] = 0;
   out_2442030683835704114[151] = 0;
   out_2442030683835704114[152] = 1;
   out_2442030683835704114[153] = 0;
   out_2442030683835704114[154] = 0;
   out_2442030683835704114[155] = 0;
   out_2442030683835704114[156] = 0;
   out_2442030683835704114[157] = 0;
   out_2442030683835704114[158] = 0;
   out_2442030683835704114[159] = 0;
   out_2442030683835704114[160] = 0;
   out_2442030683835704114[161] = 0;
   out_2442030683835704114[162] = 0;
   out_2442030683835704114[163] = 0;
   out_2442030683835704114[164] = 0;
   out_2442030683835704114[165] = 0;
   out_2442030683835704114[166] = 0;
   out_2442030683835704114[167] = 0;
   out_2442030683835704114[168] = 0;
   out_2442030683835704114[169] = 0;
   out_2442030683835704114[170] = 0;
   out_2442030683835704114[171] = 1;
   out_2442030683835704114[172] = 0;
   out_2442030683835704114[173] = 0;
   out_2442030683835704114[174] = 0;
   out_2442030683835704114[175] = 0;
   out_2442030683835704114[176] = 0;
   out_2442030683835704114[177] = 0;
   out_2442030683835704114[178] = 0;
   out_2442030683835704114[179] = 0;
   out_2442030683835704114[180] = 0;
   out_2442030683835704114[181] = 0;
   out_2442030683835704114[182] = 0;
   out_2442030683835704114[183] = 0;
   out_2442030683835704114[184] = 0;
   out_2442030683835704114[185] = 0;
   out_2442030683835704114[186] = 0;
   out_2442030683835704114[187] = 0;
   out_2442030683835704114[188] = 0;
   out_2442030683835704114[189] = 0;
   out_2442030683835704114[190] = 1;
   out_2442030683835704114[191] = 0;
   out_2442030683835704114[192] = 0;
   out_2442030683835704114[193] = 0;
   out_2442030683835704114[194] = 0;
   out_2442030683835704114[195] = 0;
   out_2442030683835704114[196] = 0;
   out_2442030683835704114[197] = 0;
   out_2442030683835704114[198] = 0;
   out_2442030683835704114[199] = 0;
   out_2442030683835704114[200] = 0;
   out_2442030683835704114[201] = 0;
   out_2442030683835704114[202] = 0;
   out_2442030683835704114[203] = 0;
   out_2442030683835704114[204] = 0;
   out_2442030683835704114[205] = 0;
   out_2442030683835704114[206] = 0;
   out_2442030683835704114[207] = 0;
   out_2442030683835704114[208] = 0;
   out_2442030683835704114[209] = 1;
   out_2442030683835704114[210] = 0;
   out_2442030683835704114[211] = 0;
   out_2442030683835704114[212] = 0;
   out_2442030683835704114[213] = 0;
   out_2442030683835704114[214] = 0;
   out_2442030683835704114[215] = 0;
   out_2442030683835704114[216] = 0;
   out_2442030683835704114[217] = 0;
   out_2442030683835704114[218] = 0;
   out_2442030683835704114[219] = 0;
   out_2442030683835704114[220] = 0;
   out_2442030683835704114[221] = 0;
   out_2442030683835704114[222] = 0;
   out_2442030683835704114[223] = 0;
   out_2442030683835704114[224] = 0;
   out_2442030683835704114[225] = 0;
   out_2442030683835704114[226] = 0;
   out_2442030683835704114[227] = 0;
   out_2442030683835704114[228] = 1;
   out_2442030683835704114[229] = 0;
   out_2442030683835704114[230] = 0;
   out_2442030683835704114[231] = 0;
   out_2442030683835704114[232] = 0;
   out_2442030683835704114[233] = 0;
   out_2442030683835704114[234] = 0;
   out_2442030683835704114[235] = 0;
   out_2442030683835704114[236] = 0;
   out_2442030683835704114[237] = 0;
   out_2442030683835704114[238] = 0;
   out_2442030683835704114[239] = 0;
   out_2442030683835704114[240] = 0;
   out_2442030683835704114[241] = 0;
   out_2442030683835704114[242] = 0;
   out_2442030683835704114[243] = 0;
   out_2442030683835704114[244] = 0;
   out_2442030683835704114[245] = 0;
   out_2442030683835704114[246] = 0;
   out_2442030683835704114[247] = 1;
   out_2442030683835704114[248] = 0;
   out_2442030683835704114[249] = 0;
   out_2442030683835704114[250] = 0;
   out_2442030683835704114[251] = 0;
   out_2442030683835704114[252] = 0;
   out_2442030683835704114[253] = 0;
   out_2442030683835704114[254] = 0;
   out_2442030683835704114[255] = 0;
   out_2442030683835704114[256] = 0;
   out_2442030683835704114[257] = 0;
   out_2442030683835704114[258] = 0;
   out_2442030683835704114[259] = 0;
   out_2442030683835704114[260] = 0;
   out_2442030683835704114[261] = 0;
   out_2442030683835704114[262] = 0;
   out_2442030683835704114[263] = 0;
   out_2442030683835704114[264] = 0;
   out_2442030683835704114[265] = 0;
   out_2442030683835704114[266] = 1;
   out_2442030683835704114[267] = 0;
   out_2442030683835704114[268] = 0;
   out_2442030683835704114[269] = 0;
   out_2442030683835704114[270] = 0;
   out_2442030683835704114[271] = 0;
   out_2442030683835704114[272] = 0;
   out_2442030683835704114[273] = 0;
   out_2442030683835704114[274] = 0;
   out_2442030683835704114[275] = 0;
   out_2442030683835704114[276] = 0;
   out_2442030683835704114[277] = 0;
   out_2442030683835704114[278] = 0;
   out_2442030683835704114[279] = 0;
   out_2442030683835704114[280] = 0;
   out_2442030683835704114[281] = 0;
   out_2442030683835704114[282] = 0;
   out_2442030683835704114[283] = 0;
   out_2442030683835704114[284] = 0;
   out_2442030683835704114[285] = 1;
   out_2442030683835704114[286] = 0;
   out_2442030683835704114[287] = 0;
   out_2442030683835704114[288] = 0;
   out_2442030683835704114[289] = 0;
   out_2442030683835704114[290] = 0;
   out_2442030683835704114[291] = 0;
   out_2442030683835704114[292] = 0;
   out_2442030683835704114[293] = 0;
   out_2442030683835704114[294] = 0;
   out_2442030683835704114[295] = 0;
   out_2442030683835704114[296] = 0;
   out_2442030683835704114[297] = 0;
   out_2442030683835704114[298] = 0;
   out_2442030683835704114[299] = 0;
   out_2442030683835704114[300] = 0;
   out_2442030683835704114[301] = 0;
   out_2442030683835704114[302] = 0;
   out_2442030683835704114[303] = 0;
   out_2442030683835704114[304] = 1;
   out_2442030683835704114[305] = 0;
   out_2442030683835704114[306] = 0;
   out_2442030683835704114[307] = 0;
   out_2442030683835704114[308] = 0;
   out_2442030683835704114[309] = 0;
   out_2442030683835704114[310] = 0;
   out_2442030683835704114[311] = 0;
   out_2442030683835704114[312] = 0;
   out_2442030683835704114[313] = 0;
   out_2442030683835704114[314] = 0;
   out_2442030683835704114[315] = 0;
   out_2442030683835704114[316] = 0;
   out_2442030683835704114[317] = 0;
   out_2442030683835704114[318] = 0;
   out_2442030683835704114[319] = 0;
   out_2442030683835704114[320] = 0;
   out_2442030683835704114[321] = 0;
   out_2442030683835704114[322] = 0;
   out_2442030683835704114[323] = 1;
}
void h_4(double *state, double *unused, double *out_5673673386662285868) {
   out_5673673386662285868[0] = state[6] + state[9];
   out_5673673386662285868[1] = state[7] + state[10];
   out_5673673386662285868[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_256490468042720012) {
   out_256490468042720012[0] = 0;
   out_256490468042720012[1] = 0;
   out_256490468042720012[2] = 0;
   out_256490468042720012[3] = 0;
   out_256490468042720012[4] = 0;
   out_256490468042720012[5] = 0;
   out_256490468042720012[6] = 1;
   out_256490468042720012[7] = 0;
   out_256490468042720012[8] = 0;
   out_256490468042720012[9] = 1;
   out_256490468042720012[10] = 0;
   out_256490468042720012[11] = 0;
   out_256490468042720012[12] = 0;
   out_256490468042720012[13] = 0;
   out_256490468042720012[14] = 0;
   out_256490468042720012[15] = 0;
   out_256490468042720012[16] = 0;
   out_256490468042720012[17] = 0;
   out_256490468042720012[18] = 0;
   out_256490468042720012[19] = 0;
   out_256490468042720012[20] = 0;
   out_256490468042720012[21] = 0;
   out_256490468042720012[22] = 0;
   out_256490468042720012[23] = 0;
   out_256490468042720012[24] = 0;
   out_256490468042720012[25] = 1;
   out_256490468042720012[26] = 0;
   out_256490468042720012[27] = 0;
   out_256490468042720012[28] = 1;
   out_256490468042720012[29] = 0;
   out_256490468042720012[30] = 0;
   out_256490468042720012[31] = 0;
   out_256490468042720012[32] = 0;
   out_256490468042720012[33] = 0;
   out_256490468042720012[34] = 0;
   out_256490468042720012[35] = 0;
   out_256490468042720012[36] = 0;
   out_256490468042720012[37] = 0;
   out_256490468042720012[38] = 0;
   out_256490468042720012[39] = 0;
   out_256490468042720012[40] = 0;
   out_256490468042720012[41] = 0;
   out_256490468042720012[42] = 0;
   out_256490468042720012[43] = 0;
   out_256490468042720012[44] = 1;
   out_256490468042720012[45] = 0;
   out_256490468042720012[46] = 0;
   out_256490468042720012[47] = 1;
   out_256490468042720012[48] = 0;
   out_256490468042720012[49] = 0;
   out_256490468042720012[50] = 0;
   out_256490468042720012[51] = 0;
   out_256490468042720012[52] = 0;
   out_256490468042720012[53] = 0;
}
void h_10(double *state, double *unused, double *out_5579523858078536433) {
   out_5579523858078536433[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_5579523858078536433[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_5579523858078536433[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_4722200315799129268) {
   out_4722200315799129268[0] = 0;
   out_4722200315799129268[1] = 9.8100000000000005*cos(state[1]);
   out_4722200315799129268[2] = 0;
   out_4722200315799129268[3] = 0;
   out_4722200315799129268[4] = -state[8];
   out_4722200315799129268[5] = state[7];
   out_4722200315799129268[6] = 0;
   out_4722200315799129268[7] = state[5];
   out_4722200315799129268[8] = -state[4];
   out_4722200315799129268[9] = 0;
   out_4722200315799129268[10] = 0;
   out_4722200315799129268[11] = 0;
   out_4722200315799129268[12] = 1;
   out_4722200315799129268[13] = 0;
   out_4722200315799129268[14] = 0;
   out_4722200315799129268[15] = 1;
   out_4722200315799129268[16] = 0;
   out_4722200315799129268[17] = 0;
   out_4722200315799129268[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_4722200315799129268[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_4722200315799129268[20] = 0;
   out_4722200315799129268[21] = state[8];
   out_4722200315799129268[22] = 0;
   out_4722200315799129268[23] = -state[6];
   out_4722200315799129268[24] = -state[5];
   out_4722200315799129268[25] = 0;
   out_4722200315799129268[26] = state[3];
   out_4722200315799129268[27] = 0;
   out_4722200315799129268[28] = 0;
   out_4722200315799129268[29] = 0;
   out_4722200315799129268[30] = 0;
   out_4722200315799129268[31] = 1;
   out_4722200315799129268[32] = 0;
   out_4722200315799129268[33] = 0;
   out_4722200315799129268[34] = 1;
   out_4722200315799129268[35] = 0;
   out_4722200315799129268[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_4722200315799129268[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_4722200315799129268[38] = 0;
   out_4722200315799129268[39] = -state[7];
   out_4722200315799129268[40] = state[6];
   out_4722200315799129268[41] = 0;
   out_4722200315799129268[42] = state[4];
   out_4722200315799129268[43] = -state[3];
   out_4722200315799129268[44] = 0;
   out_4722200315799129268[45] = 0;
   out_4722200315799129268[46] = 0;
   out_4722200315799129268[47] = 0;
   out_4722200315799129268[48] = 0;
   out_4722200315799129268[49] = 0;
   out_4722200315799129268[50] = 1;
   out_4722200315799129268[51] = 0;
   out_4722200315799129268[52] = 0;
   out_4722200315799129268[53] = 1;
}
void h_13(double *state, double *unused, double *out_3238583888242902416) {
   out_3238583888242902416[0] = state[3];
   out_3238583888242902416[1] = state[4];
   out_3238583888242902416[2] = state[5];
}
void H_13(double *state, double *unused, double *out_4090245931345244036) {
   out_4090245931345244036[0] = 0;
   out_4090245931345244036[1] = 0;
   out_4090245931345244036[2] = 0;
   out_4090245931345244036[3] = 1;
   out_4090245931345244036[4] = 0;
   out_4090245931345244036[5] = 0;
   out_4090245931345244036[6] = 0;
   out_4090245931345244036[7] = 0;
   out_4090245931345244036[8] = 0;
   out_4090245931345244036[9] = 0;
   out_4090245931345244036[10] = 0;
   out_4090245931345244036[11] = 0;
   out_4090245931345244036[12] = 0;
   out_4090245931345244036[13] = 0;
   out_4090245931345244036[14] = 0;
   out_4090245931345244036[15] = 0;
   out_4090245931345244036[16] = 0;
   out_4090245931345244036[17] = 0;
   out_4090245931345244036[18] = 0;
   out_4090245931345244036[19] = 0;
   out_4090245931345244036[20] = 0;
   out_4090245931345244036[21] = 0;
   out_4090245931345244036[22] = 1;
   out_4090245931345244036[23] = 0;
   out_4090245931345244036[24] = 0;
   out_4090245931345244036[25] = 0;
   out_4090245931345244036[26] = 0;
   out_4090245931345244036[27] = 0;
   out_4090245931345244036[28] = 0;
   out_4090245931345244036[29] = 0;
   out_4090245931345244036[30] = 0;
   out_4090245931345244036[31] = 0;
   out_4090245931345244036[32] = 0;
   out_4090245931345244036[33] = 0;
   out_4090245931345244036[34] = 0;
   out_4090245931345244036[35] = 0;
   out_4090245931345244036[36] = 0;
   out_4090245931345244036[37] = 0;
   out_4090245931345244036[38] = 0;
   out_4090245931345244036[39] = 0;
   out_4090245931345244036[40] = 0;
   out_4090245931345244036[41] = 1;
   out_4090245931345244036[42] = 0;
   out_4090245931345244036[43] = 0;
   out_4090245931345244036[44] = 0;
   out_4090245931345244036[45] = 0;
   out_4090245931345244036[46] = 0;
   out_4090245931345244036[47] = 0;
   out_4090245931345244036[48] = 0;
   out_4090245931345244036[49] = 0;
   out_4090245931345244036[50] = 0;
   out_4090245931345244036[51] = 0;
   out_4090245931345244036[52] = 0;
   out_4090245931345244036[53] = 0;
}
void h_14(double *state, double *unused, double *out_6143154130072800365) {
   out_6143154130072800365[0] = state[6];
   out_6143154130072800365[1] = state[7];
   out_6143154130072800365[2] = state[8];
}
void H_14(double *state, double *unused, double *out_3339278900338092308) {
   out_3339278900338092308[0] = 0;
   out_3339278900338092308[1] = 0;
   out_3339278900338092308[2] = 0;
   out_3339278900338092308[3] = 0;
   out_3339278900338092308[4] = 0;
   out_3339278900338092308[5] = 0;
   out_3339278900338092308[6] = 1;
   out_3339278900338092308[7] = 0;
   out_3339278900338092308[8] = 0;
   out_3339278900338092308[9] = 0;
   out_3339278900338092308[10] = 0;
   out_3339278900338092308[11] = 0;
   out_3339278900338092308[12] = 0;
   out_3339278900338092308[13] = 0;
   out_3339278900338092308[14] = 0;
   out_3339278900338092308[15] = 0;
   out_3339278900338092308[16] = 0;
   out_3339278900338092308[17] = 0;
   out_3339278900338092308[18] = 0;
   out_3339278900338092308[19] = 0;
   out_3339278900338092308[20] = 0;
   out_3339278900338092308[21] = 0;
   out_3339278900338092308[22] = 0;
   out_3339278900338092308[23] = 0;
   out_3339278900338092308[24] = 0;
   out_3339278900338092308[25] = 1;
   out_3339278900338092308[26] = 0;
   out_3339278900338092308[27] = 0;
   out_3339278900338092308[28] = 0;
   out_3339278900338092308[29] = 0;
   out_3339278900338092308[30] = 0;
   out_3339278900338092308[31] = 0;
   out_3339278900338092308[32] = 0;
   out_3339278900338092308[33] = 0;
   out_3339278900338092308[34] = 0;
   out_3339278900338092308[35] = 0;
   out_3339278900338092308[36] = 0;
   out_3339278900338092308[37] = 0;
   out_3339278900338092308[38] = 0;
   out_3339278900338092308[39] = 0;
   out_3339278900338092308[40] = 0;
   out_3339278900338092308[41] = 0;
   out_3339278900338092308[42] = 0;
   out_3339278900338092308[43] = 0;
   out_3339278900338092308[44] = 1;
   out_3339278900338092308[45] = 0;
   out_3339278900338092308[46] = 0;
   out_3339278900338092308[47] = 0;
   out_3339278900338092308[48] = 0;
   out_3339278900338092308[49] = 0;
   out_3339278900338092308[50] = 0;
   out_3339278900338092308[51] = 0;
   out_3339278900338092308[52] = 0;
   out_3339278900338092308[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_5632398071682842920) {
  err_fun(nom_x, delta_x, out_5632398071682842920);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_4793870659787952118) {
  inv_err_fun(nom_x, true_x, out_4793870659787952118);
}
void pose_H_mod_fun(double *state, double *out_4028297677886627584) {
  H_mod_fun(state, out_4028297677886627584);
}
void pose_f_fun(double *state, double dt, double *out_2980855877602976982) {
  f_fun(state,  dt, out_2980855877602976982);
}
void pose_F_fun(double *state, double dt, double *out_2442030683835704114) {
  F_fun(state,  dt, out_2442030683835704114);
}
void pose_h_4(double *state, double *unused, double *out_5673673386662285868) {
  h_4(state, unused, out_5673673386662285868);
}
void pose_H_4(double *state, double *unused, double *out_256490468042720012) {
  H_4(state, unused, out_256490468042720012);
}
void pose_h_10(double *state, double *unused, double *out_5579523858078536433) {
  h_10(state, unused, out_5579523858078536433);
}
void pose_H_10(double *state, double *unused, double *out_4722200315799129268) {
  H_10(state, unused, out_4722200315799129268);
}
void pose_h_13(double *state, double *unused, double *out_3238583888242902416) {
  h_13(state, unused, out_3238583888242902416);
}
void pose_H_13(double *state, double *unused, double *out_4090245931345244036) {
  H_13(state, unused, out_4090245931345244036);
}
void pose_h_14(double *state, double *unused, double *out_6143154130072800365) {
  h_14(state, unused, out_6143154130072800365);
}
void pose_H_14(double *state, double *unused, double *out_3339278900338092308) {
  H_14(state, unused, out_3339278900338092308);
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
