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
void err_fun(double *nom_x, double *delta_x, double *out_8199520216570231398) {
   out_8199520216570231398[0] = delta_x[0] + nom_x[0];
   out_8199520216570231398[1] = delta_x[1] + nom_x[1];
   out_8199520216570231398[2] = delta_x[2] + nom_x[2];
   out_8199520216570231398[3] = delta_x[3] + nom_x[3];
   out_8199520216570231398[4] = delta_x[4] + nom_x[4];
   out_8199520216570231398[5] = delta_x[5] + nom_x[5];
   out_8199520216570231398[6] = delta_x[6] + nom_x[6];
   out_8199520216570231398[7] = delta_x[7] + nom_x[7];
   out_8199520216570231398[8] = delta_x[8] + nom_x[8];
   out_8199520216570231398[9] = delta_x[9] + nom_x[9];
   out_8199520216570231398[10] = delta_x[10] + nom_x[10];
   out_8199520216570231398[11] = delta_x[11] + nom_x[11];
   out_8199520216570231398[12] = delta_x[12] + nom_x[12];
   out_8199520216570231398[13] = delta_x[13] + nom_x[13];
   out_8199520216570231398[14] = delta_x[14] + nom_x[14];
   out_8199520216570231398[15] = delta_x[15] + nom_x[15];
   out_8199520216570231398[16] = delta_x[16] + nom_x[16];
   out_8199520216570231398[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_6664218853987700651) {
   out_6664218853987700651[0] = -nom_x[0] + true_x[0];
   out_6664218853987700651[1] = -nom_x[1] + true_x[1];
   out_6664218853987700651[2] = -nom_x[2] + true_x[2];
   out_6664218853987700651[3] = -nom_x[3] + true_x[3];
   out_6664218853987700651[4] = -nom_x[4] + true_x[4];
   out_6664218853987700651[5] = -nom_x[5] + true_x[5];
   out_6664218853987700651[6] = -nom_x[6] + true_x[6];
   out_6664218853987700651[7] = -nom_x[7] + true_x[7];
   out_6664218853987700651[8] = -nom_x[8] + true_x[8];
   out_6664218853987700651[9] = -nom_x[9] + true_x[9];
   out_6664218853987700651[10] = -nom_x[10] + true_x[10];
   out_6664218853987700651[11] = -nom_x[11] + true_x[11];
   out_6664218853987700651[12] = -nom_x[12] + true_x[12];
   out_6664218853987700651[13] = -nom_x[13] + true_x[13];
   out_6664218853987700651[14] = -nom_x[14] + true_x[14];
   out_6664218853987700651[15] = -nom_x[15] + true_x[15];
   out_6664218853987700651[16] = -nom_x[16] + true_x[16];
   out_6664218853987700651[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_5334441709381707352) {
   out_5334441709381707352[0] = 1.0;
   out_5334441709381707352[1] = 0.0;
   out_5334441709381707352[2] = 0.0;
   out_5334441709381707352[3] = 0.0;
   out_5334441709381707352[4] = 0.0;
   out_5334441709381707352[5] = 0.0;
   out_5334441709381707352[6] = 0.0;
   out_5334441709381707352[7] = 0.0;
   out_5334441709381707352[8] = 0.0;
   out_5334441709381707352[9] = 0.0;
   out_5334441709381707352[10] = 0.0;
   out_5334441709381707352[11] = 0.0;
   out_5334441709381707352[12] = 0.0;
   out_5334441709381707352[13] = 0.0;
   out_5334441709381707352[14] = 0.0;
   out_5334441709381707352[15] = 0.0;
   out_5334441709381707352[16] = 0.0;
   out_5334441709381707352[17] = 0.0;
   out_5334441709381707352[18] = 0.0;
   out_5334441709381707352[19] = 1.0;
   out_5334441709381707352[20] = 0.0;
   out_5334441709381707352[21] = 0.0;
   out_5334441709381707352[22] = 0.0;
   out_5334441709381707352[23] = 0.0;
   out_5334441709381707352[24] = 0.0;
   out_5334441709381707352[25] = 0.0;
   out_5334441709381707352[26] = 0.0;
   out_5334441709381707352[27] = 0.0;
   out_5334441709381707352[28] = 0.0;
   out_5334441709381707352[29] = 0.0;
   out_5334441709381707352[30] = 0.0;
   out_5334441709381707352[31] = 0.0;
   out_5334441709381707352[32] = 0.0;
   out_5334441709381707352[33] = 0.0;
   out_5334441709381707352[34] = 0.0;
   out_5334441709381707352[35] = 0.0;
   out_5334441709381707352[36] = 0.0;
   out_5334441709381707352[37] = 0.0;
   out_5334441709381707352[38] = 1.0;
   out_5334441709381707352[39] = 0.0;
   out_5334441709381707352[40] = 0.0;
   out_5334441709381707352[41] = 0.0;
   out_5334441709381707352[42] = 0.0;
   out_5334441709381707352[43] = 0.0;
   out_5334441709381707352[44] = 0.0;
   out_5334441709381707352[45] = 0.0;
   out_5334441709381707352[46] = 0.0;
   out_5334441709381707352[47] = 0.0;
   out_5334441709381707352[48] = 0.0;
   out_5334441709381707352[49] = 0.0;
   out_5334441709381707352[50] = 0.0;
   out_5334441709381707352[51] = 0.0;
   out_5334441709381707352[52] = 0.0;
   out_5334441709381707352[53] = 0.0;
   out_5334441709381707352[54] = 0.0;
   out_5334441709381707352[55] = 0.0;
   out_5334441709381707352[56] = 0.0;
   out_5334441709381707352[57] = 1.0;
   out_5334441709381707352[58] = 0.0;
   out_5334441709381707352[59] = 0.0;
   out_5334441709381707352[60] = 0.0;
   out_5334441709381707352[61] = 0.0;
   out_5334441709381707352[62] = 0.0;
   out_5334441709381707352[63] = 0.0;
   out_5334441709381707352[64] = 0.0;
   out_5334441709381707352[65] = 0.0;
   out_5334441709381707352[66] = 0.0;
   out_5334441709381707352[67] = 0.0;
   out_5334441709381707352[68] = 0.0;
   out_5334441709381707352[69] = 0.0;
   out_5334441709381707352[70] = 0.0;
   out_5334441709381707352[71] = 0.0;
   out_5334441709381707352[72] = 0.0;
   out_5334441709381707352[73] = 0.0;
   out_5334441709381707352[74] = 0.0;
   out_5334441709381707352[75] = 0.0;
   out_5334441709381707352[76] = 1.0;
   out_5334441709381707352[77] = 0.0;
   out_5334441709381707352[78] = 0.0;
   out_5334441709381707352[79] = 0.0;
   out_5334441709381707352[80] = 0.0;
   out_5334441709381707352[81] = 0.0;
   out_5334441709381707352[82] = 0.0;
   out_5334441709381707352[83] = 0.0;
   out_5334441709381707352[84] = 0.0;
   out_5334441709381707352[85] = 0.0;
   out_5334441709381707352[86] = 0.0;
   out_5334441709381707352[87] = 0.0;
   out_5334441709381707352[88] = 0.0;
   out_5334441709381707352[89] = 0.0;
   out_5334441709381707352[90] = 0.0;
   out_5334441709381707352[91] = 0.0;
   out_5334441709381707352[92] = 0.0;
   out_5334441709381707352[93] = 0.0;
   out_5334441709381707352[94] = 0.0;
   out_5334441709381707352[95] = 1.0;
   out_5334441709381707352[96] = 0.0;
   out_5334441709381707352[97] = 0.0;
   out_5334441709381707352[98] = 0.0;
   out_5334441709381707352[99] = 0.0;
   out_5334441709381707352[100] = 0.0;
   out_5334441709381707352[101] = 0.0;
   out_5334441709381707352[102] = 0.0;
   out_5334441709381707352[103] = 0.0;
   out_5334441709381707352[104] = 0.0;
   out_5334441709381707352[105] = 0.0;
   out_5334441709381707352[106] = 0.0;
   out_5334441709381707352[107] = 0.0;
   out_5334441709381707352[108] = 0.0;
   out_5334441709381707352[109] = 0.0;
   out_5334441709381707352[110] = 0.0;
   out_5334441709381707352[111] = 0.0;
   out_5334441709381707352[112] = 0.0;
   out_5334441709381707352[113] = 0.0;
   out_5334441709381707352[114] = 1.0;
   out_5334441709381707352[115] = 0.0;
   out_5334441709381707352[116] = 0.0;
   out_5334441709381707352[117] = 0.0;
   out_5334441709381707352[118] = 0.0;
   out_5334441709381707352[119] = 0.0;
   out_5334441709381707352[120] = 0.0;
   out_5334441709381707352[121] = 0.0;
   out_5334441709381707352[122] = 0.0;
   out_5334441709381707352[123] = 0.0;
   out_5334441709381707352[124] = 0.0;
   out_5334441709381707352[125] = 0.0;
   out_5334441709381707352[126] = 0.0;
   out_5334441709381707352[127] = 0.0;
   out_5334441709381707352[128] = 0.0;
   out_5334441709381707352[129] = 0.0;
   out_5334441709381707352[130] = 0.0;
   out_5334441709381707352[131] = 0.0;
   out_5334441709381707352[132] = 0.0;
   out_5334441709381707352[133] = 1.0;
   out_5334441709381707352[134] = 0.0;
   out_5334441709381707352[135] = 0.0;
   out_5334441709381707352[136] = 0.0;
   out_5334441709381707352[137] = 0.0;
   out_5334441709381707352[138] = 0.0;
   out_5334441709381707352[139] = 0.0;
   out_5334441709381707352[140] = 0.0;
   out_5334441709381707352[141] = 0.0;
   out_5334441709381707352[142] = 0.0;
   out_5334441709381707352[143] = 0.0;
   out_5334441709381707352[144] = 0.0;
   out_5334441709381707352[145] = 0.0;
   out_5334441709381707352[146] = 0.0;
   out_5334441709381707352[147] = 0.0;
   out_5334441709381707352[148] = 0.0;
   out_5334441709381707352[149] = 0.0;
   out_5334441709381707352[150] = 0.0;
   out_5334441709381707352[151] = 0.0;
   out_5334441709381707352[152] = 1.0;
   out_5334441709381707352[153] = 0.0;
   out_5334441709381707352[154] = 0.0;
   out_5334441709381707352[155] = 0.0;
   out_5334441709381707352[156] = 0.0;
   out_5334441709381707352[157] = 0.0;
   out_5334441709381707352[158] = 0.0;
   out_5334441709381707352[159] = 0.0;
   out_5334441709381707352[160] = 0.0;
   out_5334441709381707352[161] = 0.0;
   out_5334441709381707352[162] = 0.0;
   out_5334441709381707352[163] = 0.0;
   out_5334441709381707352[164] = 0.0;
   out_5334441709381707352[165] = 0.0;
   out_5334441709381707352[166] = 0.0;
   out_5334441709381707352[167] = 0.0;
   out_5334441709381707352[168] = 0.0;
   out_5334441709381707352[169] = 0.0;
   out_5334441709381707352[170] = 0.0;
   out_5334441709381707352[171] = 1.0;
   out_5334441709381707352[172] = 0.0;
   out_5334441709381707352[173] = 0.0;
   out_5334441709381707352[174] = 0.0;
   out_5334441709381707352[175] = 0.0;
   out_5334441709381707352[176] = 0.0;
   out_5334441709381707352[177] = 0.0;
   out_5334441709381707352[178] = 0.0;
   out_5334441709381707352[179] = 0.0;
   out_5334441709381707352[180] = 0.0;
   out_5334441709381707352[181] = 0.0;
   out_5334441709381707352[182] = 0.0;
   out_5334441709381707352[183] = 0.0;
   out_5334441709381707352[184] = 0.0;
   out_5334441709381707352[185] = 0.0;
   out_5334441709381707352[186] = 0.0;
   out_5334441709381707352[187] = 0.0;
   out_5334441709381707352[188] = 0.0;
   out_5334441709381707352[189] = 0.0;
   out_5334441709381707352[190] = 1.0;
   out_5334441709381707352[191] = 0.0;
   out_5334441709381707352[192] = 0.0;
   out_5334441709381707352[193] = 0.0;
   out_5334441709381707352[194] = 0.0;
   out_5334441709381707352[195] = 0.0;
   out_5334441709381707352[196] = 0.0;
   out_5334441709381707352[197] = 0.0;
   out_5334441709381707352[198] = 0.0;
   out_5334441709381707352[199] = 0.0;
   out_5334441709381707352[200] = 0.0;
   out_5334441709381707352[201] = 0.0;
   out_5334441709381707352[202] = 0.0;
   out_5334441709381707352[203] = 0.0;
   out_5334441709381707352[204] = 0.0;
   out_5334441709381707352[205] = 0.0;
   out_5334441709381707352[206] = 0.0;
   out_5334441709381707352[207] = 0.0;
   out_5334441709381707352[208] = 0.0;
   out_5334441709381707352[209] = 1.0;
   out_5334441709381707352[210] = 0.0;
   out_5334441709381707352[211] = 0.0;
   out_5334441709381707352[212] = 0.0;
   out_5334441709381707352[213] = 0.0;
   out_5334441709381707352[214] = 0.0;
   out_5334441709381707352[215] = 0.0;
   out_5334441709381707352[216] = 0.0;
   out_5334441709381707352[217] = 0.0;
   out_5334441709381707352[218] = 0.0;
   out_5334441709381707352[219] = 0.0;
   out_5334441709381707352[220] = 0.0;
   out_5334441709381707352[221] = 0.0;
   out_5334441709381707352[222] = 0.0;
   out_5334441709381707352[223] = 0.0;
   out_5334441709381707352[224] = 0.0;
   out_5334441709381707352[225] = 0.0;
   out_5334441709381707352[226] = 0.0;
   out_5334441709381707352[227] = 0.0;
   out_5334441709381707352[228] = 1.0;
   out_5334441709381707352[229] = 0.0;
   out_5334441709381707352[230] = 0.0;
   out_5334441709381707352[231] = 0.0;
   out_5334441709381707352[232] = 0.0;
   out_5334441709381707352[233] = 0.0;
   out_5334441709381707352[234] = 0.0;
   out_5334441709381707352[235] = 0.0;
   out_5334441709381707352[236] = 0.0;
   out_5334441709381707352[237] = 0.0;
   out_5334441709381707352[238] = 0.0;
   out_5334441709381707352[239] = 0.0;
   out_5334441709381707352[240] = 0.0;
   out_5334441709381707352[241] = 0.0;
   out_5334441709381707352[242] = 0.0;
   out_5334441709381707352[243] = 0.0;
   out_5334441709381707352[244] = 0.0;
   out_5334441709381707352[245] = 0.0;
   out_5334441709381707352[246] = 0.0;
   out_5334441709381707352[247] = 1.0;
   out_5334441709381707352[248] = 0.0;
   out_5334441709381707352[249] = 0.0;
   out_5334441709381707352[250] = 0.0;
   out_5334441709381707352[251] = 0.0;
   out_5334441709381707352[252] = 0.0;
   out_5334441709381707352[253] = 0.0;
   out_5334441709381707352[254] = 0.0;
   out_5334441709381707352[255] = 0.0;
   out_5334441709381707352[256] = 0.0;
   out_5334441709381707352[257] = 0.0;
   out_5334441709381707352[258] = 0.0;
   out_5334441709381707352[259] = 0.0;
   out_5334441709381707352[260] = 0.0;
   out_5334441709381707352[261] = 0.0;
   out_5334441709381707352[262] = 0.0;
   out_5334441709381707352[263] = 0.0;
   out_5334441709381707352[264] = 0.0;
   out_5334441709381707352[265] = 0.0;
   out_5334441709381707352[266] = 1.0;
   out_5334441709381707352[267] = 0.0;
   out_5334441709381707352[268] = 0.0;
   out_5334441709381707352[269] = 0.0;
   out_5334441709381707352[270] = 0.0;
   out_5334441709381707352[271] = 0.0;
   out_5334441709381707352[272] = 0.0;
   out_5334441709381707352[273] = 0.0;
   out_5334441709381707352[274] = 0.0;
   out_5334441709381707352[275] = 0.0;
   out_5334441709381707352[276] = 0.0;
   out_5334441709381707352[277] = 0.0;
   out_5334441709381707352[278] = 0.0;
   out_5334441709381707352[279] = 0.0;
   out_5334441709381707352[280] = 0.0;
   out_5334441709381707352[281] = 0.0;
   out_5334441709381707352[282] = 0.0;
   out_5334441709381707352[283] = 0.0;
   out_5334441709381707352[284] = 0.0;
   out_5334441709381707352[285] = 1.0;
   out_5334441709381707352[286] = 0.0;
   out_5334441709381707352[287] = 0.0;
   out_5334441709381707352[288] = 0.0;
   out_5334441709381707352[289] = 0.0;
   out_5334441709381707352[290] = 0.0;
   out_5334441709381707352[291] = 0.0;
   out_5334441709381707352[292] = 0.0;
   out_5334441709381707352[293] = 0.0;
   out_5334441709381707352[294] = 0.0;
   out_5334441709381707352[295] = 0.0;
   out_5334441709381707352[296] = 0.0;
   out_5334441709381707352[297] = 0.0;
   out_5334441709381707352[298] = 0.0;
   out_5334441709381707352[299] = 0.0;
   out_5334441709381707352[300] = 0.0;
   out_5334441709381707352[301] = 0.0;
   out_5334441709381707352[302] = 0.0;
   out_5334441709381707352[303] = 0.0;
   out_5334441709381707352[304] = 1.0;
   out_5334441709381707352[305] = 0.0;
   out_5334441709381707352[306] = 0.0;
   out_5334441709381707352[307] = 0.0;
   out_5334441709381707352[308] = 0.0;
   out_5334441709381707352[309] = 0.0;
   out_5334441709381707352[310] = 0.0;
   out_5334441709381707352[311] = 0.0;
   out_5334441709381707352[312] = 0.0;
   out_5334441709381707352[313] = 0.0;
   out_5334441709381707352[314] = 0.0;
   out_5334441709381707352[315] = 0.0;
   out_5334441709381707352[316] = 0.0;
   out_5334441709381707352[317] = 0.0;
   out_5334441709381707352[318] = 0.0;
   out_5334441709381707352[319] = 0.0;
   out_5334441709381707352[320] = 0.0;
   out_5334441709381707352[321] = 0.0;
   out_5334441709381707352[322] = 0.0;
   out_5334441709381707352[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_5577327214882856470) {
   out_5577327214882856470[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_5577327214882856470[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_5577327214882856470[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_5577327214882856470[3] = dt*state[12] + state[3];
   out_5577327214882856470[4] = dt*state[13] + state[4];
   out_5577327214882856470[5] = dt*state[14] + state[5];
   out_5577327214882856470[6] = state[6];
   out_5577327214882856470[7] = state[7];
   out_5577327214882856470[8] = state[8];
   out_5577327214882856470[9] = state[9];
   out_5577327214882856470[10] = state[10];
   out_5577327214882856470[11] = state[11];
   out_5577327214882856470[12] = state[12];
   out_5577327214882856470[13] = state[13];
   out_5577327214882856470[14] = state[14];
   out_5577327214882856470[15] = state[15];
   out_5577327214882856470[16] = state[16];
   out_5577327214882856470[17] = state[17];
}
void F_fun(double *state, double dt, double *out_153840886414794699) {
   out_153840886414794699[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_153840886414794699[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_153840886414794699[2] = 0;
   out_153840886414794699[3] = 0;
   out_153840886414794699[4] = 0;
   out_153840886414794699[5] = 0;
   out_153840886414794699[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_153840886414794699[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_153840886414794699[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_153840886414794699[9] = 0;
   out_153840886414794699[10] = 0;
   out_153840886414794699[11] = 0;
   out_153840886414794699[12] = 0;
   out_153840886414794699[13] = 0;
   out_153840886414794699[14] = 0;
   out_153840886414794699[15] = 0;
   out_153840886414794699[16] = 0;
   out_153840886414794699[17] = 0;
   out_153840886414794699[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_153840886414794699[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_153840886414794699[20] = 0;
   out_153840886414794699[21] = 0;
   out_153840886414794699[22] = 0;
   out_153840886414794699[23] = 0;
   out_153840886414794699[24] = 0;
   out_153840886414794699[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_153840886414794699[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_153840886414794699[27] = 0;
   out_153840886414794699[28] = 0;
   out_153840886414794699[29] = 0;
   out_153840886414794699[30] = 0;
   out_153840886414794699[31] = 0;
   out_153840886414794699[32] = 0;
   out_153840886414794699[33] = 0;
   out_153840886414794699[34] = 0;
   out_153840886414794699[35] = 0;
   out_153840886414794699[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_153840886414794699[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_153840886414794699[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_153840886414794699[39] = 0;
   out_153840886414794699[40] = 0;
   out_153840886414794699[41] = 0;
   out_153840886414794699[42] = 0;
   out_153840886414794699[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_153840886414794699[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_153840886414794699[45] = 0;
   out_153840886414794699[46] = 0;
   out_153840886414794699[47] = 0;
   out_153840886414794699[48] = 0;
   out_153840886414794699[49] = 0;
   out_153840886414794699[50] = 0;
   out_153840886414794699[51] = 0;
   out_153840886414794699[52] = 0;
   out_153840886414794699[53] = 0;
   out_153840886414794699[54] = 0;
   out_153840886414794699[55] = 0;
   out_153840886414794699[56] = 0;
   out_153840886414794699[57] = 1;
   out_153840886414794699[58] = 0;
   out_153840886414794699[59] = 0;
   out_153840886414794699[60] = 0;
   out_153840886414794699[61] = 0;
   out_153840886414794699[62] = 0;
   out_153840886414794699[63] = 0;
   out_153840886414794699[64] = 0;
   out_153840886414794699[65] = 0;
   out_153840886414794699[66] = dt;
   out_153840886414794699[67] = 0;
   out_153840886414794699[68] = 0;
   out_153840886414794699[69] = 0;
   out_153840886414794699[70] = 0;
   out_153840886414794699[71] = 0;
   out_153840886414794699[72] = 0;
   out_153840886414794699[73] = 0;
   out_153840886414794699[74] = 0;
   out_153840886414794699[75] = 0;
   out_153840886414794699[76] = 1;
   out_153840886414794699[77] = 0;
   out_153840886414794699[78] = 0;
   out_153840886414794699[79] = 0;
   out_153840886414794699[80] = 0;
   out_153840886414794699[81] = 0;
   out_153840886414794699[82] = 0;
   out_153840886414794699[83] = 0;
   out_153840886414794699[84] = 0;
   out_153840886414794699[85] = dt;
   out_153840886414794699[86] = 0;
   out_153840886414794699[87] = 0;
   out_153840886414794699[88] = 0;
   out_153840886414794699[89] = 0;
   out_153840886414794699[90] = 0;
   out_153840886414794699[91] = 0;
   out_153840886414794699[92] = 0;
   out_153840886414794699[93] = 0;
   out_153840886414794699[94] = 0;
   out_153840886414794699[95] = 1;
   out_153840886414794699[96] = 0;
   out_153840886414794699[97] = 0;
   out_153840886414794699[98] = 0;
   out_153840886414794699[99] = 0;
   out_153840886414794699[100] = 0;
   out_153840886414794699[101] = 0;
   out_153840886414794699[102] = 0;
   out_153840886414794699[103] = 0;
   out_153840886414794699[104] = dt;
   out_153840886414794699[105] = 0;
   out_153840886414794699[106] = 0;
   out_153840886414794699[107] = 0;
   out_153840886414794699[108] = 0;
   out_153840886414794699[109] = 0;
   out_153840886414794699[110] = 0;
   out_153840886414794699[111] = 0;
   out_153840886414794699[112] = 0;
   out_153840886414794699[113] = 0;
   out_153840886414794699[114] = 1;
   out_153840886414794699[115] = 0;
   out_153840886414794699[116] = 0;
   out_153840886414794699[117] = 0;
   out_153840886414794699[118] = 0;
   out_153840886414794699[119] = 0;
   out_153840886414794699[120] = 0;
   out_153840886414794699[121] = 0;
   out_153840886414794699[122] = 0;
   out_153840886414794699[123] = 0;
   out_153840886414794699[124] = 0;
   out_153840886414794699[125] = 0;
   out_153840886414794699[126] = 0;
   out_153840886414794699[127] = 0;
   out_153840886414794699[128] = 0;
   out_153840886414794699[129] = 0;
   out_153840886414794699[130] = 0;
   out_153840886414794699[131] = 0;
   out_153840886414794699[132] = 0;
   out_153840886414794699[133] = 1;
   out_153840886414794699[134] = 0;
   out_153840886414794699[135] = 0;
   out_153840886414794699[136] = 0;
   out_153840886414794699[137] = 0;
   out_153840886414794699[138] = 0;
   out_153840886414794699[139] = 0;
   out_153840886414794699[140] = 0;
   out_153840886414794699[141] = 0;
   out_153840886414794699[142] = 0;
   out_153840886414794699[143] = 0;
   out_153840886414794699[144] = 0;
   out_153840886414794699[145] = 0;
   out_153840886414794699[146] = 0;
   out_153840886414794699[147] = 0;
   out_153840886414794699[148] = 0;
   out_153840886414794699[149] = 0;
   out_153840886414794699[150] = 0;
   out_153840886414794699[151] = 0;
   out_153840886414794699[152] = 1;
   out_153840886414794699[153] = 0;
   out_153840886414794699[154] = 0;
   out_153840886414794699[155] = 0;
   out_153840886414794699[156] = 0;
   out_153840886414794699[157] = 0;
   out_153840886414794699[158] = 0;
   out_153840886414794699[159] = 0;
   out_153840886414794699[160] = 0;
   out_153840886414794699[161] = 0;
   out_153840886414794699[162] = 0;
   out_153840886414794699[163] = 0;
   out_153840886414794699[164] = 0;
   out_153840886414794699[165] = 0;
   out_153840886414794699[166] = 0;
   out_153840886414794699[167] = 0;
   out_153840886414794699[168] = 0;
   out_153840886414794699[169] = 0;
   out_153840886414794699[170] = 0;
   out_153840886414794699[171] = 1;
   out_153840886414794699[172] = 0;
   out_153840886414794699[173] = 0;
   out_153840886414794699[174] = 0;
   out_153840886414794699[175] = 0;
   out_153840886414794699[176] = 0;
   out_153840886414794699[177] = 0;
   out_153840886414794699[178] = 0;
   out_153840886414794699[179] = 0;
   out_153840886414794699[180] = 0;
   out_153840886414794699[181] = 0;
   out_153840886414794699[182] = 0;
   out_153840886414794699[183] = 0;
   out_153840886414794699[184] = 0;
   out_153840886414794699[185] = 0;
   out_153840886414794699[186] = 0;
   out_153840886414794699[187] = 0;
   out_153840886414794699[188] = 0;
   out_153840886414794699[189] = 0;
   out_153840886414794699[190] = 1;
   out_153840886414794699[191] = 0;
   out_153840886414794699[192] = 0;
   out_153840886414794699[193] = 0;
   out_153840886414794699[194] = 0;
   out_153840886414794699[195] = 0;
   out_153840886414794699[196] = 0;
   out_153840886414794699[197] = 0;
   out_153840886414794699[198] = 0;
   out_153840886414794699[199] = 0;
   out_153840886414794699[200] = 0;
   out_153840886414794699[201] = 0;
   out_153840886414794699[202] = 0;
   out_153840886414794699[203] = 0;
   out_153840886414794699[204] = 0;
   out_153840886414794699[205] = 0;
   out_153840886414794699[206] = 0;
   out_153840886414794699[207] = 0;
   out_153840886414794699[208] = 0;
   out_153840886414794699[209] = 1;
   out_153840886414794699[210] = 0;
   out_153840886414794699[211] = 0;
   out_153840886414794699[212] = 0;
   out_153840886414794699[213] = 0;
   out_153840886414794699[214] = 0;
   out_153840886414794699[215] = 0;
   out_153840886414794699[216] = 0;
   out_153840886414794699[217] = 0;
   out_153840886414794699[218] = 0;
   out_153840886414794699[219] = 0;
   out_153840886414794699[220] = 0;
   out_153840886414794699[221] = 0;
   out_153840886414794699[222] = 0;
   out_153840886414794699[223] = 0;
   out_153840886414794699[224] = 0;
   out_153840886414794699[225] = 0;
   out_153840886414794699[226] = 0;
   out_153840886414794699[227] = 0;
   out_153840886414794699[228] = 1;
   out_153840886414794699[229] = 0;
   out_153840886414794699[230] = 0;
   out_153840886414794699[231] = 0;
   out_153840886414794699[232] = 0;
   out_153840886414794699[233] = 0;
   out_153840886414794699[234] = 0;
   out_153840886414794699[235] = 0;
   out_153840886414794699[236] = 0;
   out_153840886414794699[237] = 0;
   out_153840886414794699[238] = 0;
   out_153840886414794699[239] = 0;
   out_153840886414794699[240] = 0;
   out_153840886414794699[241] = 0;
   out_153840886414794699[242] = 0;
   out_153840886414794699[243] = 0;
   out_153840886414794699[244] = 0;
   out_153840886414794699[245] = 0;
   out_153840886414794699[246] = 0;
   out_153840886414794699[247] = 1;
   out_153840886414794699[248] = 0;
   out_153840886414794699[249] = 0;
   out_153840886414794699[250] = 0;
   out_153840886414794699[251] = 0;
   out_153840886414794699[252] = 0;
   out_153840886414794699[253] = 0;
   out_153840886414794699[254] = 0;
   out_153840886414794699[255] = 0;
   out_153840886414794699[256] = 0;
   out_153840886414794699[257] = 0;
   out_153840886414794699[258] = 0;
   out_153840886414794699[259] = 0;
   out_153840886414794699[260] = 0;
   out_153840886414794699[261] = 0;
   out_153840886414794699[262] = 0;
   out_153840886414794699[263] = 0;
   out_153840886414794699[264] = 0;
   out_153840886414794699[265] = 0;
   out_153840886414794699[266] = 1;
   out_153840886414794699[267] = 0;
   out_153840886414794699[268] = 0;
   out_153840886414794699[269] = 0;
   out_153840886414794699[270] = 0;
   out_153840886414794699[271] = 0;
   out_153840886414794699[272] = 0;
   out_153840886414794699[273] = 0;
   out_153840886414794699[274] = 0;
   out_153840886414794699[275] = 0;
   out_153840886414794699[276] = 0;
   out_153840886414794699[277] = 0;
   out_153840886414794699[278] = 0;
   out_153840886414794699[279] = 0;
   out_153840886414794699[280] = 0;
   out_153840886414794699[281] = 0;
   out_153840886414794699[282] = 0;
   out_153840886414794699[283] = 0;
   out_153840886414794699[284] = 0;
   out_153840886414794699[285] = 1;
   out_153840886414794699[286] = 0;
   out_153840886414794699[287] = 0;
   out_153840886414794699[288] = 0;
   out_153840886414794699[289] = 0;
   out_153840886414794699[290] = 0;
   out_153840886414794699[291] = 0;
   out_153840886414794699[292] = 0;
   out_153840886414794699[293] = 0;
   out_153840886414794699[294] = 0;
   out_153840886414794699[295] = 0;
   out_153840886414794699[296] = 0;
   out_153840886414794699[297] = 0;
   out_153840886414794699[298] = 0;
   out_153840886414794699[299] = 0;
   out_153840886414794699[300] = 0;
   out_153840886414794699[301] = 0;
   out_153840886414794699[302] = 0;
   out_153840886414794699[303] = 0;
   out_153840886414794699[304] = 1;
   out_153840886414794699[305] = 0;
   out_153840886414794699[306] = 0;
   out_153840886414794699[307] = 0;
   out_153840886414794699[308] = 0;
   out_153840886414794699[309] = 0;
   out_153840886414794699[310] = 0;
   out_153840886414794699[311] = 0;
   out_153840886414794699[312] = 0;
   out_153840886414794699[313] = 0;
   out_153840886414794699[314] = 0;
   out_153840886414794699[315] = 0;
   out_153840886414794699[316] = 0;
   out_153840886414794699[317] = 0;
   out_153840886414794699[318] = 0;
   out_153840886414794699[319] = 0;
   out_153840886414794699[320] = 0;
   out_153840886414794699[321] = 0;
   out_153840886414794699[322] = 0;
   out_153840886414794699[323] = 1;
}
void h_4(double *state, double *unused, double *out_4292041528838637925) {
   out_4292041528838637925[0] = state[6] + state[9];
   out_4292041528838637925[1] = state[7] + state[10];
   out_4292041528838637925[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_7264220557575885543) {
   out_7264220557575885543[0] = 0;
   out_7264220557575885543[1] = 0;
   out_7264220557575885543[2] = 0;
   out_7264220557575885543[3] = 0;
   out_7264220557575885543[4] = 0;
   out_7264220557575885543[5] = 0;
   out_7264220557575885543[6] = 1;
   out_7264220557575885543[7] = 0;
   out_7264220557575885543[8] = 0;
   out_7264220557575885543[9] = 1;
   out_7264220557575885543[10] = 0;
   out_7264220557575885543[11] = 0;
   out_7264220557575885543[12] = 0;
   out_7264220557575885543[13] = 0;
   out_7264220557575885543[14] = 0;
   out_7264220557575885543[15] = 0;
   out_7264220557575885543[16] = 0;
   out_7264220557575885543[17] = 0;
   out_7264220557575885543[18] = 0;
   out_7264220557575885543[19] = 0;
   out_7264220557575885543[20] = 0;
   out_7264220557575885543[21] = 0;
   out_7264220557575885543[22] = 0;
   out_7264220557575885543[23] = 0;
   out_7264220557575885543[24] = 0;
   out_7264220557575885543[25] = 1;
   out_7264220557575885543[26] = 0;
   out_7264220557575885543[27] = 0;
   out_7264220557575885543[28] = 1;
   out_7264220557575885543[29] = 0;
   out_7264220557575885543[30] = 0;
   out_7264220557575885543[31] = 0;
   out_7264220557575885543[32] = 0;
   out_7264220557575885543[33] = 0;
   out_7264220557575885543[34] = 0;
   out_7264220557575885543[35] = 0;
   out_7264220557575885543[36] = 0;
   out_7264220557575885543[37] = 0;
   out_7264220557575885543[38] = 0;
   out_7264220557575885543[39] = 0;
   out_7264220557575885543[40] = 0;
   out_7264220557575885543[41] = 0;
   out_7264220557575885543[42] = 0;
   out_7264220557575885543[43] = 0;
   out_7264220557575885543[44] = 1;
   out_7264220557575885543[45] = 0;
   out_7264220557575885543[46] = 0;
   out_7264220557575885543[47] = 1;
   out_7264220557575885543[48] = 0;
   out_7264220557575885543[49] = 0;
   out_7264220557575885543[50] = 0;
   out_7264220557575885543[51] = 0;
   out_7264220557575885543[52] = 0;
   out_7264220557575885543[53] = 0;
}
void h_10(double *state, double *unused, double *out_9179668145558624298) {
   out_9179668145558624298[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_9179668145558624298[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_9179668145558624298[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_5444952318043486315) {
   out_5444952318043486315[0] = 0;
   out_5444952318043486315[1] = 9.8100000000000005*cos(state[1]);
   out_5444952318043486315[2] = 0;
   out_5444952318043486315[3] = 0;
   out_5444952318043486315[4] = -state[8];
   out_5444952318043486315[5] = state[7];
   out_5444952318043486315[6] = 0;
   out_5444952318043486315[7] = state[5];
   out_5444952318043486315[8] = -state[4];
   out_5444952318043486315[9] = 0;
   out_5444952318043486315[10] = 0;
   out_5444952318043486315[11] = 0;
   out_5444952318043486315[12] = 1;
   out_5444952318043486315[13] = 0;
   out_5444952318043486315[14] = 0;
   out_5444952318043486315[15] = 1;
   out_5444952318043486315[16] = 0;
   out_5444952318043486315[17] = 0;
   out_5444952318043486315[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_5444952318043486315[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_5444952318043486315[20] = 0;
   out_5444952318043486315[21] = state[8];
   out_5444952318043486315[22] = 0;
   out_5444952318043486315[23] = -state[6];
   out_5444952318043486315[24] = -state[5];
   out_5444952318043486315[25] = 0;
   out_5444952318043486315[26] = state[3];
   out_5444952318043486315[27] = 0;
   out_5444952318043486315[28] = 0;
   out_5444952318043486315[29] = 0;
   out_5444952318043486315[30] = 0;
   out_5444952318043486315[31] = 1;
   out_5444952318043486315[32] = 0;
   out_5444952318043486315[33] = 0;
   out_5444952318043486315[34] = 1;
   out_5444952318043486315[35] = 0;
   out_5444952318043486315[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_5444952318043486315[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_5444952318043486315[38] = 0;
   out_5444952318043486315[39] = -state[7];
   out_5444952318043486315[40] = state[6];
   out_5444952318043486315[41] = 0;
   out_5444952318043486315[42] = state[4];
   out_5444952318043486315[43] = -state[3];
   out_5444952318043486315[44] = 0;
   out_5444952318043486315[45] = 0;
   out_5444952318043486315[46] = 0;
   out_5444952318043486315[47] = 0;
   out_5444952318043486315[48] = 0;
   out_5444952318043486315[49] = 0;
   out_5444952318043486315[50] = 1;
   out_5444952318043486315[51] = 0;
   out_5444952318043486315[52] = 0;
   out_5444952318043486315[53] = 1;
}
void h_13(double *state, double *unused, double *out_6555536983378449736) {
   out_6555536983378449736[0] = state[3];
   out_6555536983378449736[1] = state[4];
   out_6555536983378449736[2] = state[5];
}
void H_13(double *state, double *unused, double *out_7970249690801333272) {
   out_7970249690801333272[0] = 0;
   out_7970249690801333272[1] = 0;
   out_7970249690801333272[2] = 0;
   out_7970249690801333272[3] = 1;
   out_7970249690801333272[4] = 0;
   out_7970249690801333272[5] = 0;
   out_7970249690801333272[6] = 0;
   out_7970249690801333272[7] = 0;
   out_7970249690801333272[8] = 0;
   out_7970249690801333272[9] = 0;
   out_7970249690801333272[10] = 0;
   out_7970249690801333272[11] = 0;
   out_7970249690801333272[12] = 0;
   out_7970249690801333272[13] = 0;
   out_7970249690801333272[14] = 0;
   out_7970249690801333272[15] = 0;
   out_7970249690801333272[16] = 0;
   out_7970249690801333272[17] = 0;
   out_7970249690801333272[18] = 0;
   out_7970249690801333272[19] = 0;
   out_7970249690801333272[20] = 0;
   out_7970249690801333272[21] = 0;
   out_7970249690801333272[22] = 1;
   out_7970249690801333272[23] = 0;
   out_7970249690801333272[24] = 0;
   out_7970249690801333272[25] = 0;
   out_7970249690801333272[26] = 0;
   out_7970249690801333272[27] = 0;
   out_7970249690801333272[28] = 0;
   out_7970249690801333272[29] = 0;
   out_7970249690801333272[30] = 0;
   out_7970249690801333272[31] = 0;
   out_7970249690801333272[32] = 0;
   out_7970249690801333272[33] = 0;
   out_7970249690801333272[34] = 0;
   out_7970249690801333272[35] = 0;
   out_7970249690801333272[36] = 0;
   out_7970249690801333272[37] = 0;
   out_7970249690801333272[38] = 0;
   out_7970249690801333272[39] = 0;
   out_7970249690801333272[40] = 0;
   out_7970249690801333272[41] = 1;
   out_7970249690801333272[42] = 0;
   out_7970249690801333272[43] = 0;
   out_7970249690801333272[44] = 0;
   out_7970249690801333272[45] = 0;
   out_7970249690801333272[46] = 0;
   out_7970249690801333272[47] = 0;
   out_7970249690801333272[48] = 0;
   out_7970249690801333272[49] = 0;
   out_7970249690801333272[50] = 0;
   out_7970249690801333272[51] = 0;
   out_7970249690801333272[52] = 0;
   out_7970249690801333272[53] = 0;
}
void h_14(double *state, double *unused, double *out_7794980382073332925) {
   out_7794980382073332925[0] = state[6];
   out_7794980382073332925[1] = state[7];
   out_7794980382073332925[2] = state[8];
}
void H_14(double *state, double *unused, double *out_7219282659794181544) {
   out_7219282659794181544[0] = 0;
   out_7219282659794181544[1] = 0;
   out_7219282659794181544[2] = 0;
   out_7219282659794181544[3] = 0;
   out_7219282659794181544[4] = 0;
   out_7219282659794181544[5] = 0;
   out_7219282659794181544[6] = 1;
   out_7219282659794181544[7] = 0;
   out_7219282659794181544[8] = 0;
   out_7219282659794181544[9] = 0;
   out_7219282659794181544[10] = 0;
   out_7219282659794181544[11] = 0;
   out_7219282659794181544[12] = 0;
   out_7219282659794181544[13] = 0;
   out_7219282659794181544[14] = 0;
   out_7219282659794181544[15] = 0;
   out_7219282659794181544[16] = 0;
   out_7219282659794181544[17] = 0;
   out_7219282659794181544[18] = 0;
   out_7219282659794181544[19] = 0;
   out_7219282659794181544[20] = 0;
   out_7219282659794181544[21] = 0;
   out_7219282659794181544[22] = 0;
   out_7219282659794181544[23] = 0;
   out_7219282659794181544[24] = 0;
   out_7219282659794181544[25] = 1;
   out_7219282659794181544[26] = 0;
   out_7219282659794181544[27] = 0;
   out_7219282659794181544[28] = 0;
   out_7219282659794181544[29] = 0;
   out_7219282659794181544[30] = 0;
   out_7219282659794181544[31] = 0;
   out_7219282659794181544[32] = 0;
   out_7219282659794181544[33] = 0;
   out_7219282659794181544[34] = 0;
   out_7219282659794181544[35] = 0;
   out_7219282659794181544[36] = 0;
   out_7219282659794181544[37] = 0;
   out_7219282659794181544[38] = 0;
   out_7219282659794181544[39] = 0;
   out_7219282659794181544[40] = 0;
   out_7219282659794181544[41] = 0;
   out_7219282659794181544[42] = 0;
   out_7219282659794181544[43] = 0;
   out_7219282659794181544[44] = 1;
   out_7219282659794181544[45] = 0;
   out_7219282659794181544[46] = 0;
   out_7219282659794181544[47] = 0;
   out_7219282659794181544[48] = 0;
   out_7219282659794181544[49] = 0;
   out_7219282659794181544[50] = 0;
   out_7219282659794181544[51] = 0;
   out_7219282659794181544[52] = 0;
   out_7219282659794181544[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_8199520216570231398) {
  err_fun(nom_x, delta_x, out_8199520216570231398);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_6664218853987700651) {
  inv_err_fun(nom_x, true_x, out_6664218853987700651);
}
void pose_H_mod_fun(double *state, double *out_5334441709381707352) {
  H_mod_fun(state, out_5334441709381707352);
}
void pose_f_fun(double *state, double dt, double *out_5577327214882856470) {
  f_fun(state,  dt, out_5577327214882856470);
}
void pose_F_fun(double *state, double dt, double *out_153840886414794699) {
  F_fun(state,  dt, out_153840886414794699);
}
void pose_h_4(double *state, double *unused, double *out_4292041528838637925) {
  h_4(state, unused, out_4292041528838637925);
}
void pose_H_4(double *state, double *unused, double *out_7264220557575885543) {
  H_4(state, unused, out_7264220557575885543);
}
void pose_h_10(double *state, double *unused, double *out_9179668145558624298) {
  h_10(state, unused, out_9179668145558624298);
}
void pose_H_10(double *state, double *unused, double *out_5444952318043486315) {
  H_10(state, unused, out_5444952318043486315);
}
void pose_h_13(double *state, double *unused, double *out_6555536983378449736) {
  h_13(state, unused, out_6555536983378449736);
}
void pose_H_13(double *state, double *unused, double *out_7970249690801333272) {
  H_13(state, unused, out_7970249690801333272);
}
void pose_h_14(double *state, double *unused, double *out_7794980382073332925) {
  h_14(state, unused, out_7794980382073332925);
}
void pose_H_14(double *state, double *unused, double *out_7219282659794181544) {
  H_14(state, unused, out_7219282659794181544);
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
