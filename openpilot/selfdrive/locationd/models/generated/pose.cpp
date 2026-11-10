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
void err_fun(double *nom_x, double *delta_x, double *out_5957606077413925934) {
   out_5957606077413925934[0] = delta_x[0] + nom_x[0];
   out_5957606077413925934[1] = delta_x[1] + nom_x[1];
   out_5957606077413925934[2] = delta_x[2] + nom_x[2];
   out_5957606077413925934[3] = delta_x[3] + nom_x[3];
   out_5957606077413925934[4] = delta_x[4] + nom_x[4];
   out_5957606077413925934[5] = delta_x[5] + nom_x[5];
   out_5957606077413925934[6] = delta_x[6] + nom_x[6];
   out_5957606077413925934[7] = delta_x[7] + nom_x[7];
   out_5957606077413925934[8] = delta_x[8] + nom_x[8];
   out_5957606077413925934[9] = delta_x[9] + nom_x[9];
   out_5957606077413925934[10] = delta_x[10] + nom_x[10];
   out_5957606077413925934[11] = delta_x[11] + nom_x[11];
   out_5957606077413925934[12] = delta_x[12] + nom_x[12];
   out_5957606077413925934[13] = delta_x[13] + nom_x[13];
   out_5957606077413925934[14] = delta_x[14] + nom_x[14];
   out_5957606077413925934[15] = delta_x[15] + nom_x[15];
   out_5957606077413925934[16] = delta_x[16] + nom_x[16];
   out_5957606077413925934[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_1181409950769297795) {
   out_1181409950769297795[0] = -nom_x[0] + true_x[0];
   out_1181409950769297795[1] = -nom_x[1] + true_x[1];
   out_1181409950769297795[2] = -nom_x[2] + true_x[2];
   out_1181409950769297795[3] = -nom_x[3] + true_x[3];
   out_1181409950769297795[4] = -nom_x[4] + true_x[4];
   out_1181409950769297795[5] = -nom_x[5] + true_x[5];
   out_1181409950769297795[6] = -nom_x[6] + true_x[6];
   out_1181409950769297795[7] = -nom_x[7] + true_x[7];
   out_1181409950769297795[8] = -nom_x[8] + true_x[8];
   out_1181409950769297795[9] = -nom_x[9] + true_x[9];
   out_1181409950769297795[10] = -nom_x[10] + true_x[10];
   out_1181409950769297795[11] = -nom_x[11] + true_x[11];
   out_1181409950769297795[12] = -nom_x[12] + true_x[12];
   out_1181409950769297795[13] = -nom_x[13] + true_x[13];
   out_1181409950769297795[14] = -nom_x[14] + true_x[14];
   out_1181409950769297795[15] = -nom_x[15] + true_x[15];
   out_1181409950769297795[16] = -nom_x[16] + true_x[16];
   out_1181409950769297795[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_1576273921813987430) {
   out_1576273921813987430[0] = 1.0;
   out_1576273921813987430[1] = 0.0;
   out_1576273921813987430[2] = 0.0;
   out_1576273921813987430[3] = 0.0;
   out_1576273921813987430[4] = 0.0;
   out_1576273921813987430[5] = 0.0;
   out_1576273921813987430[6] = 0.0;
   out_1576273921813987430[7] = 0.0;
   out_1576273921813987430[8] = 0.0;
   out_1576273921813987430[9] = 0.0;
   out_1576273921813987430[10] = 0.0;
   out_1576273921813987430[11] = 0.0;
   out_1576273921813987430[12] = 0.0;
   out_1576273921813987430[13] = 0.0;
   out_1576273921813987430[14] = 0.0;
   out_1576273921813987430[15] = 0.0;
   out_1576273921813987430[16] = 0.0;
   out_1576273921813987430[17] = 0.0;
   out_1576273921813987430[18] = 0.0;
   out_1576273921813987430[19] = 1.0;
   out_1576273921813987430[20] = 0.0;
   out_1576273921813987430[21] = 0.0;
   out_1576273921813987430[22] = 0.0;
   out_1576273921813987430[23] = 0.0;
   out_1576273921813987430[24] = 0.0;
   out_1576273921813987430[25] = 0.0;
   out_1576273921813987430[26] = 0.0;
   out_1576273921813987430[27] = 0.0;
   out_1576273921813987430[28] = 0.0;
   out_1576273921813987430[29] = 0.0;
   out_1576273921813987430[30] = 0.0;
   out_1576273921813987430[31] = 0.0;
   out_1576273921813987430[32] = 0.0;
   out_1576273921813987430[33] = 0.0;
   out_1576273921813987430[34] = 0.0;
   out_1576273921813987430[35] = 0.0;
   out_1576273921813987430[36] = 0.0;
   out_1576273921813987430[37] = 0.0;
   out_1576273921813987430[38] = 1.0;
   out_1576273921813987430[39] = 0.0;
   out_1576273921813987430[40] = 0.0;
   out_1576273921813987430[41] = 0.0;
   out_1576273921813987430[42] = 0.0;
   out_1576273921813987430[43] = 0.0;
   out_1576273921813987430[44] = 0.0;
   out_1576273921813987430[45] = 0.0;
   out_1576273921813987430[46] = 0.0;
   out_1576273921813987430[47] = 0.0;
   out_1576273921813987430[48] = 0.0;
   out_1576273921813987430[49] = 0.0;
   out_1576273921813987430[50] = 0.0;
   out_1576273921813987430[51] = 0.0;
   out_1576273921813987430[52] = 0.0;
   out_1576273921813987430[53] = 0.0;
   out_1576273921813987430[54] = 0.0;
   out_1576273921813987430[55] = 0.0;
   out_1576273921813987430[56] = 0.0;
   out_1576273921813987430[57] = 1.0;
   out_1576273921813987430[58] = 0.0;
   out_1576273921813987430[59] = 0.0;
   out_1576273921813987430[60] = 0.0;
   out_1576273921813987430[61] = 0.0;
   out_1576273921813987430[62] = 0.0;
   out_1576273921813987430[63] = 0.0;
   out_1576273921813987430[64] = 0.0;
   out_1576273921813987430[65] = 0.0;
   out_1576273921813987430[66] = 0.0;
   out_1576273921813987430[67] = 0.0;
   out_1576273921813987430[68] = 0.0;
   out_1576273921813987430[69] = 0.0;
   out_1576273921813987430[70] = 0.0;
   out_1576273921813987430[71] = 0.0;
   out_1576273921813987430[72] = 0.0;
   out_1576273921813987430[73] = 0.0;
   out_1576273921813987430[74] = 0.0;
   out_1576273921813987430[75] = 0.0;
   out_1576273921813987430[76] = 1.0;
   out_1576273921813987430[77] = 0.0;
   out_1576273921813987430[78] = 0.0;
   out_1576273921813987430[79] = 0.0;
   out_1576273921813987430[80] = 0.0;
   out_1576273921813987430[81] = 0.0;
   out_1576273921813987430[82] = 0.0;
   out_1576273921813987430[83] = 0.0;
   out_1576273921813987430[84] = 0.0;
   out_1576273921813987430[85] = 0.0;
   out_1576273921813987430[86] = 0.0;
   out_1576273921813987430[87] = 0.0;
   out_1576273921813987430[88] = 0.0;
   out_1576273921813987430[89] = 0.0;
   out_1576273921813987430[90] = 0.0;
   out_1576273921813987430[91] = 0.0;
   out_1576273921813987430[92] = 0.0;
   out_1576273921813987430[93] = 0.0;
   out_1576273921813987430[94] = 0.0;
   out_1576273921813987430[95] = 1.0;
   out_1576273921813987430[96] = 0.0;
   out_1576273921813987430[97] = 0.0;
   out_1576273921813987430[98] = 0.0;
   out_1576273921813987430[99] = 0.0;
   out_1576273921813987430[100] = 0.0;
   out_1576273921813987430[101] = 0.0;
   out_1576273921813987430[102] = 0.0;
   out_1576273921813987430[103] = 0.0;
   out_1576273921813987430[104] = 0.0;
   out_1576273921813987430[105] = 0.0;
   out_1576273921813987430[106] = 0.0;
   out_1576273921813987430[107] = 0.0;
   out_1576273921813987430[108] = 0.0;
   out_1576273921813987430[109] = 0.0;
   out_1576273921813987430[110] = 0.0;
   out_1576273921813987430[111] = 0.0;
   out_1576273921813987430[112] = 0.0;
   out_1576273921813987430[113] = 0.0;
   out_1576273921813987430[114] = 1.0;
   out_1576273921813987430[115] = 0.0;
   out_1576273921813987430[116] = 0.0;
   out_1576273921813987430[117] = 0.0;
   out_1576273921813987430[118] = 0.0;
   out_1576273921813987430[119] = 0.0;
   out_1576273921813987430[120] = 0.0;
   out_1576273921813987430[121] = 0.0;
   out_1576273921813987430[122] = 0.0;
   out_1576273921813987430[123] = 0.0;
   out_1576273921813987430[124] = 0.0;
   out_1576273921813987430[125] = 0.0;
   out_1576273921813987430[126] = 0.0;
   out_1576273921813987430[127] = 0.0;
   out_1576273921813987430[128] = 0.0;
   out_1576273921813987430[129] = 0.0;
   out_1576273921813987430[130] = 0.0;
   out_1576273921813987430[131] = 0.0;
   out_1576273921813987430[132] = 0.0;
   out_1576273921813987430[133] = 1.0;
   out_1576273921813987430[134] = 0.0;
   out_1576273921813987430[135] = 0.0;
   out_1576273921813987430[136] = 0.0;
   out_1576273921813987430[137] = 0.0;
   out_1576273921813987430[138] = 0.0;
   out_1576273921813987430[139] = 0.0;
   out_1576273921813987430[140] = 0.0;
   out_1576273921813987430[141] = 0.0;
   out_1576273921813987430[142] = 0.0;
   out_1576273921813987430[143] = 0.0;
   out_1576273921813987430[144] = 0.0;
   out_1576273921813987430[145] = 0.0;
   out_1576273921813987430[146] = 0.0;
   out_1576273921813987430[147] = 0.0;
   out_1576273921813987430[148] = 0.0;
   out_1576273921813987430[149] = 0.0;
   out_1576273921813987430[150] = 0.0;
   out_1576273921813987430[151] = 0.0;
   out_1576273921813987430[152] = 1.0;
   out_1576273921813987430[153] = 0.0;
   out_1576273921813987430[154] = 0.0;
   out_1576273921813987430[155] = 0.0;
   out_1576273921813987430[156] = 0.0;
   out_1576273921813987430[157] = 0.0;
   out_1576273921813987430[158] = 0.0;
   out_1576273921813987430[159] = 0.0;
   out_1576273921813987430[160] = 0.0;
   out_1576273921813987430[161] = 0.0;
   out_1576273921813987430[162] = 0.0;
   out_1576273921813987430[163] = 0.0;
   out_1576273921813987430[164] = 0.0;
   out_1576273921813987430[165] = 0.0;
   out_1576273921813987430[166] = 0.0;
   out_1576273921813987430[167] = 0.0;
   out_1576273921813987430[168] = 0.0;
   out_1576273921813987430[169] = 0.0;
   out_1576273921813987430[170] = 0.0;
   out_1576273921813987430[171] = 1.0;
   out_1576273921813987430[172] = 0.0;
   out_1576273921813987430[173] = 0.0;
   out_1576273921813987430[174] = 0.0;
   out_1576273921813987430[175] = 0.0;
   out_1576273921813987430[176] = 0.0;
   out_1576273921813987430[177] = 0.0;
   out_1576273921813987430[178] = 0.0;
   out_1576273921813987430[179] = 0.0;
   out_1576273921813987430[180] = 0.0;
   out_1576273921813987430[181] = 0.0;
   out_1576273921813987430[182] = 0.0;
   out_1576273921813987430[183] = 0.0;
   out_1576273921813987430[184] = 0.0;
   out_1576273921813987430[185] = 0.0;
   out_1576273921813987430[186] = 0.0;
   out_1576273921813987430[187] = 0.0;
   out_1576273921813987430[188] = 0.0;
   out_1576273921813987430[189] = 0.0;
   out_1576273921813987430[190] = 1.0;
   out_1576273921813987430[191] = 0.0;
   out_1576273921813987430[192] = 0.0;
   out_1576273921813987430[193] = 0.0;
   out_1576273921813987430[194] = 0.0;
   out_1576273921813987430[195] = 0.0;
   out_1576273921813987430[196] = 0.0;
   out_1576273921813987430[197] = 0.0;
   out_1576273921813987430[198] = 0.0;
   out_1576273921813987430[199] = 0.0;
   out_1576273921813987430[200] = 0.0;
   out_1576273921813987430[201] = 0.0;
   out_1576273921813987430[202] = 0.0;
   out_1576273921813987430[203] = 0.0;
   out_1576273921813987430[204] = 0.0;
   out_1576273921813987430[205] = 0.0;
   out_1576273921813987430[206] = 0.0;
   out_1576273921813987430[207] = 0.0;
   out_1576273921813987430[208] = 0.0;
   out_1576273921813987430[209] = 1.0;
   out_1576273921813987430[210] = 0.0;
   out_1576273921813987430[211] = 0.0;
   out_1576273921813987430[212] = 0.0;
   out_1576273921813987430[213] = 0.0;
   out_1576273921813987430[214] = 0.0;
   out_1576273921813987430[215] = 0.0;
   out_1576273921813987430[216] = 0.0;
   out_1576273921813987430[217] = 0.0;
   out_1576273921813987430[218] = 0.0;
   out_1576273921813987430[219] = 0.0;
   out_1576273921813987430[220] = 0.0;
   out_1576273921813987430[221] = 0.0;
   out_1576273921813987430[222] = 0.0;
   out_1576273921813987430[223] = 0.0;
   out_1576273921813987430[224] = 0.0;
   out_1576273921813987430[225] = 0.0;
   out_1576273921813987430[226] = 0.0;
   out_1576273921813987430[227] = 0.0;
   out_1576273921813987430[228] = 1.0;
   out_1576273921813987430[229] = 0.0;
   out_1576273921813987430[230] = 0.0;
   out_1576273921813987430[231] = 0.0;
   out_1576273921813987430[232] = 0.0;
   out_1576273921813987430[233] = 0.0;
   out_1576273921813987430[234] = 0.0;
   out_1576273921813987430[235] = 0.0;
   out_1576273921813987430[236] = 0.0;
   out_1576273921813987430[237] = 0.0;
   out_1576273921813987430[238] = 0.0;
   out_1576273921813987430[239] = 0.0;
   out_1576273921813987430[240] = 0.0;
   out_1576273921813987430[241] = 0.0;
   out_1576273921813987430[242] = 0.0;
   out_1576273921813987430[243] = 0.0;
   out_1576273921813987430[244] = 0.0;
   out_1576273921813987430[245] = 0.0;
   out_1576273921813987430[246] = 0.0;
   out_1576273921813987430[247] = 1.0;
   out_1576273921813987430[248] = 0.0;
   out_1576273921813987430[249] = 0.0;
   out_1576273921813987430[250] = 0.0;
   out_1576273921813987430[251] = 0.0;
   out_1576273921813987430[252] = 0.0;
   out_1576273921813987430[253] = 0.0;
   out_1576273921813987430[254] = 0.0;
   out_1576273921813987430[255] = 0.0;
   out_1576273921813987430[256] = 0.0;
   out_1576273921813987430[257] = 0.0;
   out_1576273921813987430[258] = 0.0;
   out_1576273921813987430[259] = 0.0;
   out_1576273921813987430[260] = 0.0;
   out_1576273921813987430[261] = 0.0;
   out_1576273921813987430[262] = 0.0;
   out_1576273921813987430[263] = 0.0;
   out_1576273921813987430[264] = 0.0;
   out_1576273921813987430[265] = 0.0;
   out_1576273921813987430[266] = 1.0;
   out_1576273921813987430[267] = 0.0;
   out_1576273921813987430[268] = 0.0;
   out_1576273921813987430[269] = 0.0;
   out_1576273921813987430[270] = 0.0;
   out_1576273921813987430[271] = 0.0;
   out_1576273921813987430[272] = 0.0;
   out_1576273921813987430[273] = 0.0;
   out_1576273921813987430[274] = 0.0;
   out_1576273921813987430[275] = 0.0;
   out_1576273921813987430[276] = 0.0;
   out_1576273921813987430[277] = 0.0;
   out_1576273921813987430[278] = 0.0;
   out_1576273921813987430[279] = 0.0;
   out_1576273921813987430[280] = 0.0;
   out_1576273921813987430[281] = 0.0;
   out_1576273921813987430[282] = 0.0;
   out_1576273921813987430[283] = 0.0;
   out_1576273921813987430[284] = 0.0;
   out_1576273921813987430[285] = 1.0;
   out_1576273921813987430[286] = 0.0;
   out_1576273921813987430[287] = 0.0;
   out_1576273921813987430[288] = 0.0;
   out_1576273921813987430[289] = 0.0;
   out_1576273921813987430[290] = 0.0;
   out_1576273921813987430[291] = 0.0;
   out_1576273921813987430[292] = 0.0;
   out_1576273921813987430[293] = 0.0;
   out_1576273921813987430[294] = 0.0;
   out_1576273921813987430[295] = 0.0;
   out_1576273921813987430[296] = 0.0;
   out_1576273921813987430[297] = 0.0;
   out_1576273921813987430[298] = 0.0;
   out_1576273921813987430[299] = 0.0;
   out_1576273921813987430[300] = 0.0;
   out_1576273921813987430[301] = 0.0;
   out_1576273921813987430[302] = 0.0;
   out_1576273921813987430[303] = 0.0;
   out_1576273921813987430[304] = 1.0;
   out_1576273921813987430[305] = 0.0;
   out_1576273921813987430[306] = 0.0;
   out_1576273921813987430[307] = 0.0;
   out_1576273921813987430[308] = 0.0;
   out_1576273921813987430[309] = 0.0;
   out_1576273921813987430[310] = 0.0;
   out_1576273921813987430[311] = 0.0;
   out_1576273921813987430[312] = 0.0;
   out_1576273921813987430[313] = 0.0;
   out_1576273921813987430[314] = 0.0;
   out_1576273921813987430[315] = 0.0;
   out_1576273921813987430[316] = 0.0;
   out_1576273921813987430[317] = 0.0;
   out_1576273921813987430[318] = 0.0;
   out_1576273921813987430[319] = 0.0;
   out_1576273921813987430[320] = 0.0;
   out_1576273921813987430[321] = 0.0;
   out_1576273921813987430[322] = 0.0;
   out_1576273921813987430[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_8480623208419059652) {
   out_8480623208419059652[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_8480623208419059652[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_8480623208419059652[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_8480623208419059652[3] = dt*state[12] + state[3];
   out_8480623208419059652[4] = dt*state[13] + state[4];
   out_8480623208419059652[5] = dt*state[14] + state[5];
   out_8480623208419059652[6] = state[6];
   out_8480623208419059652[7] = state[7];
   out_8480623208419059652[8] = state[8];
   out_8480623208419059652[9] = state[9];
   out_8480623208419059652[10] = state[10];
   out_8480623208419059652[11] = state[11];
   out_8480623208419059652[12] = state[12];
   out_8480623208419059652[13] = state[13];
   out_8480623208419059652[14] = state[14];
   out_8480623208419059652[15] = state[15];
   out_8480623208419059652[16] = state[16];
   out_8480623208419059652[17] = state[17];
}
void F_fun(double *state, double dt, double *out_6948763257431598150) {
   out_6948763257431598150[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6948763257431598150[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6948763257431598150[2] = 0;
   out_6948763257431598150[3] = 0;
   out_6948763257431598150[4] = 0;
   out_6948763257431598150[5] = 0;
   out_6948763257431598150[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6948763257431598150[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6948763257431598150[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_6948763257431598150[9] = 0;
   out_6948763257431598150[10] = 0;
   out_6948763257431598150[11] = 0;
   out_6948763257431598150[12] = 0;
   out_6948763257431598150[13] = 0;
   out_6948763257431598150[14] = 0;
   out_6948763257431598150[15] = 0;
   out_6948763257431598150[16] = 0;
   out_6948763257431598150[17] = 0;
   out_6948763257431598150[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6948763257431598150[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6948763257431598150[20] = 0;
   out_6948763257431598150[21] = 0;
   out_6948763257431598150[22] = 0;
   out_6948763257431598150[23] = 0;
   out_6948763257431598150[24] = 0;
   out_6948763257431598150[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6948763257431598150[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_6948763257431598150[27] = 0;
   out_6948763257431598150[28] = 0;
   out_6948763257431598150[29] = 0;
   out_6948763257431598150[30] = 0;
   out_6948763257431598150[31] = 0;
   out_6948763257431598150[32] = 0;
   out_6948763257431598150[33] = 0;
   out_6948763257431598150[34] = 0;
   out_6948763257431598150[35] = 0;
   out_6948763257431598150[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6948763257431598150[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6948763257431598150[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6948763257431598150[39] = 0;
   out_6948763257431598150[40] = 0;
   out_6948763257431598150[41] = 0;
   out_6948763257431598150[42] = 0;
   out_6948763257431598150[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6948763257431598150[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_6948763257431598150[45] = 0;
   out_6948763257431598150[46] = 0;
   out_6948763257431598150[47] = 0;
   out_6948763257431598150[48] = 0;
   out_6948763257431598150[49] = 0;
   out_6948763257431598150[50] = 0;
   out_6948763257431598150[51] = 0;
   out_6948763257431598150[52] = 0;
   out_6948763257431598150[53] = 0;
   out_6948763257431598150[54] = 0;
   out_6948763257431598150[55] = 0;
   out_6948763257431598150[56] = 0;
   out_6948763257431598150[57] = 1;
   out_6948763257431598150[58] = 0;
   out_6948763257431598150[59] = 0;
   out_6948763257431598150[60] = 0;
   out_6948763257431598150[61] = 0;
   out_6948763257431598150[62] = 0;
   out_6948763257431598150[63] = 0;
   out_6948763257431598150[64] = 0;
   out_6948763257431598150[65] = 0;
   out_6948763257431598150[66] = dt;
   out_6948763257431598150[67] = 0;
   out_6948763257431598150[68] = 0;
   out_6948763257431598150[69] = 0;
   out_6948763257431598150[70] = 0;
   out_6948763257431598150[71] = 0;
   out_6948763257431598150[72] = 0;
   out_6948763257431598150[73] = 0;
   out_6948763257431598150[74] = 0;
   out_6948763257431598150[75] = 0;
   out_6948763257431598150[76] = 1;
   out_6948763257431598150[77] = 0;
   out_6948763257431598150[78] = 0;
   out_6948763257431598150[79] = 0;
   out_6948763257431598150[80] = 0;
   out_6948763257431598150[81] = 0;
   out_6948763257431598150[82] = 0;
   out_6948763257431598150[83] = 0;
   out_6948763257431598150[84] = 0;
   out_6948763257431598150[85] = dt;
   out_6948763257431598150[86] = 0;
   out_6948763257431598150[87] = 0;
   out_6948763257431598150[88] = 0;
   out_6948763257431598150[89] = 0;
   out_6948763257431598150[90] = 0;
   out_6948763257431598150[91] = 0;
   out_6948763257431598150[92] = 0;
   out_6948763257431598150[93] = 0;
   out_6948763257431598150[94] = 0;
   out_6948763257431598150[95] = 1;
   out_6948763257431598150[96] = 0;
   out_6948763257431598150[97] = 0;
   out_6948763257431598150[98] = 0;
   out_6948763257431598150[99] = 0;
   out_6948763257431598150[100] = 0;
   out_6948763257431598150[101] = 0;
   out_6948763257431598150[102] = 0;
   out_6948763257431598150[103] = 0;
   out_6948763257431598150[104] = dt;
   out_6948763257431598150[105] = 0;
   out_6948763257431598150[106] = 0;
   out_6948763257431598150[107] = 0;
   out_6948763257431598150[108] = 0;
   out_6948763257431598150[109] = 0;
   out_6948763257431598150[110] = 0;
   out_6948763257431598150[111] = 0;
   out_6948763257431598150[112] = 0;
   out_6948763257431598150[113] = 0;
   out_6948763257431598150[114] = 1;
   out_6948763257431598150[115] = 0;
   out_6948763257431598150[116] = 0;
   out_6948763257431598150[117] = 0;
   out_6948763257431598150[118] = 0;
   out_6948763257431598150[119] = 0;
   out_6948763257431598150[120] = 0;
   out_6948763257431598150[121] = 0;
   out_6948763257431598150[122] = 0;
   out_6948763257431598150[123] = 0;
   out_6948763257431598150[124] = 0;
   out_6948763257431598150[125] = 0;
   out_6948763257431598150[126] = 0;
   out_6948763257431598150[127] = 0;
   out_6948763257431598150[128] = 0;
   out_6948763257431598150[129] = 0;
   out_6948763257431598150[130] = 0;
   out_6948763257431598150[131] = 0;
   out_6948763257431598150[132] = 0;
   out_6948763257431598150[133] = 1;
   out_6948763257431598150[134] = 0;
   out_6948763257431598150[135] = 0;
   out_6948763257431598150[136] = 0;
   out_6948763257431598150[137] = 0;
   out_6948763257431598150[138] = 0;
   out_6948763257431598150[139] = 0;
   out_6948763257431598150[140] = 0;
   out_6948763257431598150[141] = 0;
   out_6948763257431598150[142] = 0;
   out_6948763257431598150[143] = 0;
   out_6948763257431598150[144] = 0;
   out_6948763257431598150[145] = 0;
   out_6948763257431598150[146] = 0;
   out_6948763257431598150[147] = 0;
   out_6948763257431598150[148] = 0;
   out_6948763257431598150[149] = 0;
   out_6948763257431598150[150] = 0;
   out_6948763257431598150[151] = 0;
   out_6948763257431598150[152] = 1;
   out_6948763257431598150[153] = 0;
   out_6948763257431598150[154] = 0;
   out_6948763257431598150[155] = 0;
   out_6948763257431598150[156] = 0;
   out_6948763257431598150[157] = 0;
   out_6948763257431598150[158] = 0;
   out_6948763257431598150[159] = 0;
   out_6948763257431598150[160] = 0;
   out_6948763257431598150[161] = 0;
   out_6948763257431598150[162] = 0;
   out_6948763257431598150[163] = 0;
   out_6948763257431598150[164] = 0;
   out_6948763257431598150[165] = 0;
   out_6948763257431598150[166] = 0;
   out_6948763257431598150[167] = 0;
   out_6948763257431598150[168] = 0;
   out_6948763257431598150[169] = 0;
   out_6948763257431598150[170] = 0;
   out_6948763257431598150[171] = 1;
   out_6948763257431598150[172] = 0;
   out_6948763257431598150[173] = 0;
   out_6948763257431598150[174] = 0;
   out_6948763257431598150[175] = 0;
   out_6948763257431598150[176] = 0;
   out_6948763257431598150[177] = 0;
   out_6948763257431598150[178] = 0;
   out_6948763257431598150[179] = 0;
   out_6948763257431598150[180] = 0;
   out_6948763257431598150[181] = 0;
   out_6948763257431598150[182] = 0;
   out_6948763257431598150[183] = 0;
   out_6948763257431598150[184] = 0;
   out_6948763257431598150[185] = 0;
   out_6948763257431598150[186] = 0;
   out_6948763257431598150[187] = 0;
   out_6948763257431598150[188] = 0;
   out_6948763257431598150[189] = 0;
   out_6948763257431598150[190] = 1;
   out_6948763257431598150[191] = 0;
   out_6948763257431598150[192] = 0;
   out_6948763257431598150[193] = 0;
   out_6948763257431598150[194] = 0;
   out_6948763257431598150[195] = 0;
   out_6948763257431598150[196] = 0;
   out_6948763257431598150[197] = 0;
   out_6948763257431598150[198] = 0;
   out_6948763257431598150[199] = 0;
   out_6948763257431598150[200] = 0;
   out_6948763257431598150[201] = 0;
   out_6948763257431598150[202] = 0;
   out_6948763257431598150[203] = 0;
   out_6948763257431598150[204] = 0;
   out_6948763257431598150[205] = 0;
   out_6948763257431598150[206] = 0;
   out_6948763257431598150[207] = 0;
   out_6948763257431598150[208] = 0;
   out_6948763257431598150[209] = 1;
   out_6948763257431598150[210] = 0;
   out_6948763257431598150[211] = 0;
   out_6948763257431598150[212] = 0;
   out_6948763257431598150[213] = 0;
   out_6948763257431598150[214] = 0;
   out_6948763257431598150[215] = 0;
   out_6948763257431598150[216] = 0;
   out_6948763257431598150[217] = 0;
   out_6948763257431598150[218] = 0;
   out_6948763257431598150[219] = 0;
   out_6948763257431598150[220] = 0;
   out_6948763257431598150[221] = 0;
   out_6948763257431598150[222] = 0;
   out_6948763257431598150[223] = 0;
   out_6948763257431598150[224] = 0;
   out_6948763257431598150[225] = 0;
   out_6948763257431598150[226] = 0;
   out_6948763257431598150[227] = 0;
   out_6948763257431598150[228] = 1;
   out_6948763257431598150[229] = 0;
   out_6948763257431598150[230] = 0;
   out_6948763257431598150[231] = 0;
   out_6948763257431598150[232] = 0;
   out_6948763257431598150[233] = 0;
   out_6948763257431598150[234] = 0;
   out_6948763257431598150[235] = 0;
   out_6948763257431598150[236] = 0;
   out_6948763257431598150[237] = 0;
   out_6948763257431598150[238] = 0;
   out_6948763257431598150[239] = 0;
   out_6948763257431598150[240] = 0;
   out_6948763257431598150[241] = 0;
   out_6948763257431598150[242] = 0;
   out_6948763257431598150[243] = 0;
   out_6948763257431598150[244] = 0;
   out_6948763257431598150[245] = 0;
   out_6948763257431598150[246] = 0;
   out_6948763257431598150[247] = 1;
   out_6948763257431598150[248] = 0;
   out_6948763257431598150[249] = 0;
   out_6948763257431598150[250] = 0;
   out_6948763257431598150[251] = 0;
   out_6948763257431598150[252] = 0;
   out_6948763257431598150[253] = 0;
   out_6948763257431598150[254] = 0;
   out_6948763257431598150[255] = 0;
   out_6948763257431598150[256] = 0;
   out_6948763257431598150[257] = 0;
   out_6948763257431598150[258] = 0;
   out_6948763257431598150[259] = 0;
   out_6948763257431598150[260] = 0;
   out_6948763257431598150[261] = 0;
   out_6948763257431598150[262] = 0;
   out_6948763257431598150[263] = 0;
   out_6948763257431598150[264] = 0;
   out_6948763257431598150[265] = 0;
   out_6948763257431598150[266] = 1;
   out_6948763257431598150[267] = 0;
   out_6948763257431598150[268] = 0;
   out_6948763257431598150[269] = 0;
   out_6948763257431598150[270] = 0;
   out_6948763257431598150[271] = 0;
   out_6948763257431598150[272] = 0;
   out_6948763257431598150[273] = 0;
   out_6948763257431598150[274] = 0;
   out_6948763257431598150[275] = 0;
   out_6948763257431598150[276] = 0;
   out_6948763257431598150[277] = 0;
   out_6948763257431598150[278] = 0;
   out_6948763257431598150[279] = 0;
   out_6948763257431598150[280] = 0;
   out_6948763257431598150[281] = 0;
   out_6948763257431598150[282] = 0;
   out_6948763257431598150[283] = 0;
   out_6948763257431598150[284] = 0;
   out_6948763257431598150[285] = 1;
   out_6948763257431598150[286] = 0;
   out_6948763257431598150[287] = 0;
   out_6948763257431598150[288] = 0;
   out_6948763257431598150[289] = 0;
   out_6948763257431598150[290] = 0;
   out_6948763257431598150[291] = 0;
   out_6948763257431598150[292] = 0;
   out_6948763257431598150[293] = 0;
   out_6948763257431598150[294] = 0;
   out_6948763257431598150[295] = 0;
   out_6948763257431598150[296] = 0;
   out_6948763257431598150[297] = 0;
   out_6948763257431598150[298] = 0;
   out_6948763257431598150[299] = 0;
   out_6948763257431598150[300] = 0;
   out_6948763257431598150[301] = 0;
   out_6948763257431598150[302] = 0;
   out_6948763257431598150[303] = 0;
   out_6948763257431598150[304] = 1;
   out_6948763257431598150[305] = 0;
   out_6948763257431598150[306] = 0;
   out_6948763257431598150[307] = 0;
   out_6948763257431598150[308] = 0;
   out_6948763257431598150[309] = 0;
   out_6948763257431598150[310] = 0;
   out_6948763257431598150[311] = 0;
   out_6948763257431598150[312] = 0;
   out_6948763257431598150[313] = 0;
   out_6948763257431598150[314] = 0;
   out_6948763257431598150[315] = 0;
   out_6948763257431598150[316] = 0;
   out_6948763257431598150[317] = 0;
   out_6948763257431598150[318] = 0;
   out_6948763257431598150[319] = 0;
   out_6948763257431598150[320] = 0;
   out_6948763257431598150[321] = 0;
   out_6948763257431598150[322] = 0;
   out_6948763257431598150[323] = 1;
}
void h_4(double *state, double *unused, double *out_7022920791114964455) {
   out_7022920791114964455[0] = state[6] + state[9];
   out_7022920791114964455[1] = state[7] + state[10];
   out_7022920791114964455[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_2330434923868700037) {
   out_2330434923868700037[0] = 0;
   out_2330434923868700037[1] = 0;
   out_2330434923868700037[2] = 0;
   out_2330434923868700037[3] = 0;
   out_2330434923868700037[4] = 0;
   out_2330434923868700037[5] = 0;
   out_2330434923868700037[6] = 1;
   out_2330434923868700037[7] = 0;
   out_2330434923868700037[8] = 0;
   out_2330434923868700037[9] = 1;
   out_2330434923868700037[10] = 0;
   out_2330434923868700037[11] = 0;
   out_2330434923868700037[12] = 0;
   out_2330434923868700037[13] = 0;
   out_2330434923868700037[14] = 0;
   out_2330434923868700037[15] = 0;
   out_2330434923868700037[16] = 0;
   out_2330434923868700037[17] = 0;
   out_2330434923868700037[18] = 0;
   out_2330434923868700037[19] = 0;
   out_2330434923868700037[20] = 0;
   out_2330434923868700037[21] = 0;
   out_2330434923868700037[22] = 0;
   out_2330434923868700037[23] = 0;
   out_2330434923868700037[24] = 0;
   out_2330434923868700037[25] = 1;
   out_2330434923868700037[26] = 0;
   out_2330434923868700037[27] = 0;
   out_2330434923868700037[28] = 1;
   out_2330434923868700037[29] = 0;
   out_2330434923868700037[30] = 0;
   out_2330434923868700037[31] = 0;
   out_2330434923868700037[32] = 0;
   out_2330434923868700037[33] = 0;
   out_2330434923868700037[34] = 0;
   out_2330434923868700037[35] = 0;
   out_2330434923868700037[36] = 0;
   out_2330434923868700037[37] = 0;
   out_2330434923868700037[38] = 0;
   out_2330434923868700037[39] = 0;
   out_2330434923868700037[40] = 0;
   out_2330434923868700037[41] = 0;
   out_2330434923868700037[42] = 0;
   out_2330434923868700037[43] = 0;
   out_2330434923868700037[44] = 1;
   out_2330434923868700037[45] = 0;
   out_2330434923868700037[46] = 0;
   out_2330434923868700037[47] = 1;
   out_2330434923868700037[48] = 0;
   out_2330434923868700037[49] = 0;
   out_2330434923868700037[50] = 0;
   out_2330434923868700037[51] = 0;
   out_2330434923868700037[52] = 0;
   out_2330434923868700037[53] = 0;
}
void h_10(double *state, double *unused, double *out_8675686222808342194) {
   out_8675686222808342194[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_8675686222808342194[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_8675686222808342194[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_638794999717561462) {
   out_638794999717561462[0] = 0;
   out_638794999717561462[1] = 9.8100000000000005*cos(state[1]);
   out_638794999717561462[2] = 0;
   out_638794999717561462[3] = 0;
   out_638794999717561462[4] = -state[8];
   out_638794999717561462[5] = state[7];
   out_638794999717561462[6] = 0;
   out_638794999717561462[7] = state[5];
   out_638794999717561462[8] = -state[4];
   out_638794999717561462[9] = 0;
   out_638794999717561462[10] = 0;
   out_638794999717561462[11] = 0;
   out_638794999717561462[12] = 1;
   out_638794999717561462[13] = 0;
   out_638794999717561462[14] = 0;
   out_638794999717561462[15] = 1;
   out_638794999717561462[16] = 0;
   out_638794999717561462[17] = 0;
   out_638794999717561462[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_638794999717561462[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_638794999717561462[20] = 0;
   out_638794999717561462[21] = state[8];
   out_638794999717561462[22] = 0;
   out_638794999717561462[23] = -state[6];
   out_638794999717561462[24] = -state[5];
   out_638794999717561462[25] = 0;
   out_638794999717561462[26] = state[3];
   out_638794999717561462[27] = 0;
   out_638794999717561462[28] = 0;
   out_638794999717561462[29] = 0;
   out_638794999717561462[30] = 0;
   out_638794999717561462[31] = 1;
   out_638794999717561462[32] = 0;
   out_638794999717561462[33] = 0;
   out_638794999717561462[34] = 1;
   out_638794999717561462[35] = 0;
   out_638794999717561462[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_638794999717561462[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_638794999717561462[38] = 0;
   out_638794999717561462[39] = -state[7];
   out_638794999717561462[40] = state[6];
   out_638794999717561462[41] = 0;
   out_638794999717561462[42] = state[4];
   out_638794999717561462[43] = -state[3];
   out_638794999717561462[44] = 0;
   out_638794999717561462[45] = 0;
   out_638794999717561462[46] = 0;
   out_638794999717561462[47] = 0;
   out_638794999717561462[48] = 0;
   out_638794999717561462[49] = 0;
   out_638794999717561462[50] = 1;
   out_638794999717561462[51] = 0;
   out_638794999717561462[52] = 0;
   out_638794999717561462[53] = 1;
}
void h_13(double *state, double *unused, double *out_4089366875480109516) {
   out_4089366875480109516[0] = state[3];
   out_4089366875480109516[1] = state[4];
   out_4089366875480109516[2] = state[5];
}
void H_13(double *state, double *unused, double *out_5542708749201032838) {
   out_5542708749201032838[0] = 0;
   out_5542708749201032838[1] = 0;
   out_5542708749201032838[2] = 0;
   out_5542708749201032838[3] = 1;
   out_5542708749201032838[4] = 0;
   out_5542708749201032838[5] = 0;
   out_5542708749201032838[6] = 0;
   out_5542708749201032838[7] = 0;
   out_5542708749201032838[8] = 0;
   out_5542708749201032838[9] = 0;
   out_5542708749201032838[10] = 0;
   out_5542708749201032838[11] = 0;
   out_5542708749201032838[12] = 0;
   out_5542708749201032838[13] = 0;
   out_5542708749201032838[14] = 0;
   out_5542708749201032838[15] = 0;
   out_5542708749201032838[16] = 0;
   out_5542708749201032838[17] = 0;
   out_5542708749201032838[18] = 0;
   out_5542708749201032838[19] = 0;
   out_5542708749201032838[20] = 0;
   out_5542708749201032838[21] = 0;
   out_5542708749201032838[22] = 1;
   out_5542708749201032838[23] = 0;
   out_5542708749201032838[24] = 0;
   out_5542708749201032838[25] = 0;
   out_5542708749201032838[26] = 0;
   out_5542708749201032838[27] = 0;
   out_5542708749201032838[28] = 0;
   out_5542708749201032838[29] = 0;
   out_5542708749201032838[30] = 0;
   out_5542708749201032838[31] = 0;
   out_5542708749201032838[32] = 0;
   out_5542708749201032838[33] = 0;
   out_5542708749201032838[34] = 0;
   out_5542708749201032838[35] = 0;
   out_5542708749201032838[36] = 0;
   out_5542708749201032838[37] = 0;
   out_5542708749201032838[38] = 0;
   out_5542708749201032838[39] = 0;
   out_5542708749201032838[40] = 0;
   out_5542708749201032838[41] = 1;
   out_5542708749201032838[42] = 0;
   out_5542708749201032838[43] = 0;
   out_5542708749201032838[44] = 0;
   out_5542708749201032838[45] = 0;
   out_5542708749201032838[46] = 0;
   out_5542708749201032838[47] = 0;
   out_5542708749201032838[48] = 0;
   out_5542708749201032838[49] = 0;
   out_5542708749201032838[50] = 0;
   out_5542708749201032838[51] = 0;
   out_5542708749201032838[52] = 0;
   out_5542708749201032838[53] = 0;
}
void h_14(double *state, double *unused, double *out_7178822489175268675) {
   out_7178822489175268675[0] = state[6];
   out_7178822489175268675[1] = state[7];
   out_7178822489175268675[2] = state[8];
}
void H_14(double *state, double *unused, double *out_6293675780208184566) {
   out_6293675780208184566[0] = 0;
   out_6293675780208184566[1] = 0;
   out_6293675780208184566[2] = 0;
   out_6293675780208184566[3] = 0;
   out_6293675780208184566[4] = 0;
   out_6293675780208184566[5] = 0;
   out_6293675780208184566[6] = 1;
   out_6293675780208184566[7] = 0;
   out_6293675780208184566[8] = 0;
   out_6293675780208184566[9] = 0;
   out_6293675780208184566[10] = 0;
   out_6293675780208184566[11] = 0;
   out_6293675780208184566[12] = 0;
   out_6293675780208184566[13] = 0;
   out_6293675780208184566[14] = 0;
   out_6293675780208184566[15] = 0;
   out_6293675780208184566[16] = 0;
   out_6293675780208184566[17] = 0;
   out_6293675780208184566[18] = 0;
   out_6293675780208184566[19] = 0;
   out_6293675780208184566[20] = 0;
   out_6293675780208184566[21] = 0;
   out_6293675780208184566[22] = 0;
   out_6293675780208184566[23] = 0;
   out_6293675780208184566[24] = 0;
   out_6293675780208184566[25] = 1;
   out_6293675780208184566[26] = 0;
   out_6293675780208184566[27] = 0;
   out_6293675780208184566[28] = 0;
   out_6293675780208184566[29] = 0;
   out_6293675780208184566[30] = 0;
   out_6293675780208184566[31] = 0;
   out_6293675780208184566[32] = 0;
   out_6293675780208184566[33] = 0;
   out_6293675780208184566[34] = 0;
   out_6293675780208184566[35] = 0;
   out_6293675780208184566[36] = 0;
   out_6293675780208184566[37] = 0;
   out_6293675780208184566[38] = 0;
   out_6293675780208184566[39] = 0;
   out_6293675780208184566[40] = 0;
   out_6293675780208184566[41] = 0;
   out_6293675780208184566[42] = 0;
   out_6293675780208184566[43] = 0;
   out_6293675780208184566[44] = 1;
   out_6293675780208184566[45] = 0;
   out_6293675780208184566[46] = 0;
   out_6293675780208184566[47] = 0;
   out_6293675780208184566[48] = 0;
   out_6293675780208184566[49] = 0;
   out_6293675780208184566[50] = 0;
   out_6293675780208184566[51] = 0;
   out_6293675780208184566[52] = 0;
   out_6293675780208184566[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_5957606077413925934) {
  err_fun(nom_x, delta_x, out_5957606077413925934);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_1181409950769297795) {
  inv_err_fun(nom_x, true_x, out_1181409950769297795);
}
void pose_H_mod_fun(double *state, double *out_1576273921813987430) {
  H_mod_fun(state, out_1576273921813987430);
}
void pose_f_fun(double *state, double dt, double *out_8480623208419059652) {
  f_fun(state,  dt, out_8480623208419059652);
}
void pose_F_fun(double *state, double dt, double *out_6948763257431598150) {
  F_fun(state,  dt, out_6948763257431598150);
}
void pose_h_4(double *state, double *unused, double *out_7022920791114964455) {
  h_4(state, unused, out_7022920791114964455);
}
void pose_H_4(double *state, double *unused, double *out_2330434923868700037) {
  H_4(state, unused, out_2330434923868700037);
}
void pose_h_10(double *state, double *unused, double *out_8675686222808342194) {
  h_10(state, unused, out_8675686222808342194);
}
void pose_H_10(double *state, double *unused, double *out_638794999717561462) {
  H_10(state, unused, out_638794999717561462);
}
void pose_h_13(double *state, double *unused, double *out_4089366875480109516) {
  h_13(state, unused, out_4089366875480109516);
}
void pose_H_13(double *state, double *unused, double *out_5542708749201032838) {
  H_13(state, unused, out_5542708749201032838);
}
void pose_h_14(double *state, double *unused, double *out_7178822489175268675) {
  h_14(state, unused, out_7178822489175268675);
}
void pose_H_14(double *state, double *unused, double *out_6293675780208184566) {
  H_14(state, unused, out_6293675780208184566);
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
