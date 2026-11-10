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
void err_fun(double *nom_x, double *delta_x, double *out_2016014972386344778) {
   out_2016014972386344778[0] = delta_x[0] + nom_x[0];
   out_2016014972386344778[1] = delta_x[1] + nom_x[1];
   out_2016014972386344778[2] = delta_x[2] + nom_x[2];
   out_2016014972386344778[3] = delta_x[3] + nom_x[3];
   out_2016014972386344778[4] = delta_x[4] + nom_x[4];
   out_2016014972386344778[5] = delta_x[5] + nom_x[5];
   out_2016014972386344778[6] = delta_x[6] + nom_x[6];
   out_2016014972386344778[7] = delta_x[7] + nom_x[7];
   out_2016014972386344778[8] = delta_x[8] + nom_x[8];
   out_2016014972386344778[9] = delta_x[9] + nom_x[9];
   out_2016014972386344778[10] = delta_x[10] + nom_x[10];
   out_2016014972386344778[11] = delta_x[11] + nom_x[11];
   out_2016014972386344778[12] = delta_x[12] + nom_x[12];
   out_2016014972386344778[13] = delta_x[13] + nom_x[13];
   out_2016014972386344778[14] = delta_x[14] + nom_x[14];
   out_2016014972386344778[15] = delta_x[15] + nom_x[15];
   out_2016014972386344778[16] = delta_x[16] + nom_x[16];
   out_2016014972386344778[17] = delta_x[17] + nom_x[17];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_7065715320158728118) {
   out_7065715320158728118[0] = -nom_x[0] + true_x[0];
   out_7065715320158728118[1] = -nom_x[1] + true_x[1];
   out_7065715320158728118[2] = -nom_x[2] + true_x[2];
   out_7065715320158728118[3] = -nom_x[3] + true_x[3];
   out_7065715320158728118[4] = -nom_x[4] + true_x[4];
   out_7065715320158728118[5] = -nom_x[5] + true_x[5];
   out_7065715320158728118[6] = -nom_x[6] + true_x[6];
   out_7065715320158728118[7] = -nom_x[7] + true_x[7];
   out_7065715320158728118[8] = -nom_x[8] + true_x[8];
   out_7065715320158728118[9] = -nom_x[9] + true_x[9];
   out_7065715320158728118[10] = -nom_x[10] + true_x[10];
   out_7065715320158728118[11] = -nom_x[11] + true_x[11];
   out_7065715320158728118[12] = -nom_x[12] + true_x[12];
   out_7065715320158728118[13] = -nom_x[13] + true_x[13];
   out_7065715320158728118[14] = -nom_x[14] + true_x[14];
   out_7065715320158728118[15] = -nom_x[15] + true_x[15];
   out_7065715320158728118[16] = -nom_x[16] + true_x[16];
   out_7065715320158728118[17] = -nom_x[17] + true_x[17];
}
void H_mod_fun(double *state, double *out_515136171011412265) {
   out_515136171011412265[0] = 1.0;
   out_515136171011412265[1] = 0.0;
   out_515136171011412265[2] = 0.0;
   out_515136171011412265[3] = 0.0;
   out_515136171011412265[4] = 0.0;
   out_515136171011412265[5] = 0.0;
   out_515136171011412265[6] = 0.0;
   out_515136171011412265[7] = 0.0;
   out_515136171011412265[8] = 0.0;
   out_515136171011412265[9] = 0.0;
   out_515136171011412265[10] = 0.0;
   out_515136171011412265[11] = 0.0;
   out_515136171011412265[12] = 0.0;
   out_515136171011412265[13] = 0.0;
   out_515136171011412265[14] = 0.0;
   out_515136171011412265[15] = 0.0;
   out_515136171011412265[16] = 0.0;
   out_515136171011412265[17] = 0.0;
   out_515136171011412265[18] = 0.0;
   out_515136171011412265[19] = 1.0;
   out_515136171011412265[20] = 0.0;
   out_515136171011412265[21] = 0.0;
   out_515136171011412265[22] = 0.0;
   out_515136171011412265[23] = 0.0;
   out_515136171011412265[24] = 0.0;
   out_515136171011412265[25] = 0.0;
   out_515136171011412265[26] = 0.0;
   out_515136171011412265[27] = 0.0;
   out_515136171011412265[28] = 0.0;
   out_515136171011412265[29] = 0.0;
   out_515136171011412265[30] = 0.0;
   out_515136171011412265[31] = 0.0;
   out_515136171011412265[32] = 0.0;
   out_515136171011412265[33] = 0.0;
   out_515136171011412265[34] = 0.0;
   out_515136171011412265[35] = 0.0;
   out_515136171011412265[36] = 0.0;
   out_515136171011412265[37] = 0.0;
   out_515136171011412265[38] = 1.0;
   out_515136171011412265[39] = 0.0;
   out_515136171011412265[40] = 0.0;
   out_515136171011412265[41] = 0.0;
   out_515136171011412265[42] = 0.0;
   out_515136171011412265[43] = 0.0;
   out_515136171011412265[44] = 0.0;
   out_515136171011412265[45] = 0.0;
   out_515136171011412265[46] = 0.0;
   out_515136171011412265[47] = 0.0;
   out_515136171011412265[48] = 0.0;
   out_515136171011412265[49] = 0.0;
   out_515136171011412265[50] = 0.0;
   out_515136171011412265[51] = 0.0;
   out_515136171011412265[52] = 0.0;
   out_515136171011412265[53] = 0.0;
   out_515136171011412265[54] = 0.0;
   out_515136171011412265[55] = 0.0;
   out_515136171011412265[56] = 0.0;
   out_515136171011412265[57] = 1.0;
   out_515136171011412265[58] = 0.0;
   out_515136171011412265[59] = 0.0;
   out_515136171011412265[60] = 0.0;
   out_515136171011412265[61] = 0.0;
   out_515136171011412265[62] = 0.0;
   out_515136171011412265[63] = 0.0;
   out_515136171011412265[64] = 0.0;
   out_515136171011412265[65] = 0.0;
   out_515136171011412265[66] = 0.0;
   out_515136171011412265[67] = 0.0;
   out_515136171011412265[68] = 0.0;
   out_515136171011412265[69] = 0.0;
   out_515136171011412265[70] = 0.0;
   out_515136171011412265[71] = 0.0;
   out_515136171011412265[72] = 0.0;
   out_515136171011412265[73] = 0.0;
   out_515136171011412265[74] = 0.0;
   out_515136171011412265[75] = 0.0;
   out_515136171011412265[76] = 1.0;
   out_515136171011412265[77] = 0.0;
   out_515136171011412265[78] = 0.0;
   out_515136171011412265[79] = 0.0;
   out_515136171011412265[80] = 0.0;
   out_515136171011412265[81] = 0.0;
   out_515136171011412265[82] = 0.0;
   out_515136171011412265[83] = 0.0;
   out_515136171011412265[84] = 0.0;
   out_515136171011412265[85] = 0.0;
   out_515136171011412265[86] = 0.0;
   out_515136171011412265[87] = 0.0;
   out_515136171011412265[88] = 0.0;
   out_515136171011412265[89] = 0.0;
   out_515136171011412265[90] = 0.0;
   out_515136171011412265[91] = 0.0;
   out_515136171011412265[92] = 0.0;
   out_515136171011412265[93] = 0.0;
   out_515136171011412265[94] = 0.0;
   out_515136171011412265[95] = 1.0;
   out_515136171011412265[96] = 0.0;
   out_515136171011412265[97] = 0.0;
   out_515136171011412265[98] = 0.0;
   out_515136171011412265[99] = 0.0;
   out_515136171011412265[100] = 0.0;
   out_515136171011412265[101] = 0.0;
   out_515136171011412265[102] = 0.0;
   out_515136171011412265[103] = 0.0;
   out_515136171011412265[104] = 0.0;
   out_515136171011412265[105] = 0.0;
   out_515136171011412265[106] = 0.0;
   out_515136171011412265[107] = 0.0;
   out_515136171011412265[108] = 0.0;
   out_515136171011412265[109] = 0.0;
   out_515136171011412265[110] = 0.0;
   out_515136171011412265[111] = 0.0;
   out_515136171011412265[112] = 0.0;
   out_515136171011412265[113] = 0.0;
   out_515136171011412265[114] = 1.0;
   out_515136171011412265[115] = 0.0;
   out_515136171011412265[116] = 0.0;
   out_515136171011412265[117] = 0.0;
   out_515136171011412265[118] = 0.0;
   out_515136171011412265[119] = 0.0;
   out_515136171011412265[120] = 0.0;
   out_515136171011412265[121] = 0.0;
   out_515136171011412265[122] = 0.0;
   out_515136171011412265[123] = 0.0;
   out_515136171011412265[124] = 0.0;
   out_515136171011412265[125] = 0.0;
   out_515136171011412265[126] = 0.0;
   out_515136171011412265[127] = 0.0;
   out_515136171011412265[128] = 0.0;
   out_515136171011412265[129] = 0.0;
   out_515136171011412265[130] = 0.0;
   out_515136171011412265[131] = 0.0;
   out_515136171011412265[132] = 0.0;
   out_515136171011412265[133] = 1.0;
   out_515136171011412265[134] = 0.0;
   out_515136171011412265[135] = 0.0;
   out_515136171011412265[136] = 0.0;
   out_515136171011412265[137] = 0.0;
   out_515136171011412265[138] = 0.0;
   out_515136171011412265[139] = 0.0;
   out_515136171011412265[140] = 0.0;
   out_515136171011412265[141] = 0.0;
   out_515136171011412265[142] = 0.0;
   out_515136171011412265[143] = 0.0;
   out_515136171011412265[144] = 0.0;
   out_515136171011412265[145] = 0.0;
   out_515136171011412265[146] = 0.0;
   out_515136171011412265[147] = 0.0;
   out_515136171011412265[148] = 0.0;
   out_515136171011412265[149] = 0.0;
   out_515136171011412265[150] = 0.0;
   out_515136171011412265[151] = 0.0;
   out_515136171011412265[152] = 1.0;
   out_515136171011412265[153] = 0.0;
   out_515136171011412265[154] = 0.0;
   out_515136171011412265[155] = 0.0;
   out_515136171011412265[156] = 0.0;
   out_515136171011412265[157] = 0.0;
   out_515136171011412265[158] = 0.0;
   out_515136171011412265[159] = 0.0;
   out_515136171011412265[160] = 0.0;
   out_515136171011412265[161] = 0.0;
   out_515136171011412265[162] = 0.0;
   out_515136171011412265[163] = 0.0;
   out_515136171011412265[164] = 0.0;
   out_515136171011412265[165] = 0.0;
   out_515136171011412265[166] = 0.0;
   out_515136171011412265[167] = 0.0;
   out_515136171011412265[168] = 0.0;
   out_515136171011412265[169] = 0.0;
   out_515136171011412265[170] = 0.0;
   out_515136171011412265[171] = 1.0;
   out_515136171011412265[172] = 0.0;
   out_515136171011412265[173] = 0.0;
   out_515136171011412265[174] = 0.0;
   out_515136171011412265[175] = 0.0;
   out_515136171011412265[176] = 0.0;
   out_515136171011412265[177] = 0.0;
   out_515136171011412265[178] = 0.0;
   out_515136171011412265[179] = 0.0;
   out_515136171011412265[180] = 0.0;
   out_515136171011412265[181] = 0.0;
   out_515136171011412265[182] = 0.0;
   out_515136171011412265[183] = 0.0;
   out_515136171011412265[184] = 0.0;
   out_515136171011412265[185] = 0.0;
   out_515136171011412265[186] = 0.0;
   out_515136171011412265[187] = 0.0;
   out_515136171011412265[188] = 0.0;
   out_515136171011412265[189] = 0.0;
   out_515136171011412265[190] = 1.0;
   out_515136171011412265[191] = 0.0;
   out_515136171011412265[192] = 0.0;
   out_515136171011412265[193] = 0.0;
   out_515136171011412265[194] = 0.0;
   out_515136171011412265[195] = 0.0;
   out_515136171011412265[196] = 0.0;
   out_515136171011412265[197] = 0.0;
   out_515136171011412265[198] = 0.0;
   out_515136171011412265[199] = 0.0;
   out_515136171011412265[200] = 0.0;
   out_515136171011412265[201] = 0.0;
   out_515136171011412265[202] = 0.0;
   out_515136171011412265[203] = 0.0;
   out_515136171011412265[204] = 0.0;
   out_515136171011412265[205] = 0.0;
   out_515136171011412265[206] = 0.0;
   out_515136171011412265[207] = 0.0;
   out_515136171011412265[208] = 0.0;
   out_515136171011412265[209] = 1.0;
   out_515136171011412265[210] = 0.0;
   out_515136171011412265[211] = 0.0;
   out_515136171011412265[212] = 0.0;
   out_515136171011412265[213] = 0.0;
   out_515136171011412265[214] = 0.0;
   out_515136171011412265[215] = 0.0;
   out_515136171011412265[216] = 0.0;
   out_515136171011412265[217] = 0.0;
   out_515136171011412265[218] = 0.0;
   out_515136171011412265[219] = 0.0;
   out_515136171011412265[220] = 0.0;
   out_515136171011412265[221] = 0.0;
   out_515136171011412265[222] = 0.0;
   out_515136171011412265[223] = 0.0;
   out_515136171011412265[224] = 0.0;
   out_515136171011412265[225] = 0.0;
   out_515136171011412265[226] = 0.0;
   out_515136171011412265[227] = 0.0;
   out_515136171011412265[228] = 1.0;
   out_515136171011412265[229] = 0.0;
   out_515136171011412265[230] = 0.0;
   out_515136171011412265[231] = 0.0;
   out_515136171011412265[232] = 0.0;
   out_515136171011412265[233] = 0.0;
   out_515136171011412265[234] = 0.0;
   out_515136171011412265[235] = 0.0;
   out_515136171011412265[236] = 0.0;
   out_515136171011412265[237] = 0.0;
   out_515136171011412265[238] = 0.0;
   out_515136171011412265[239] = 0.0;
   out_515136171011412265[240] = 0.0;
   out_515136171011412265[241] = 0.0;
   out_515136171011412265[242] = 0.0;
   out_515136171011412265[243] = 0.0;
   out_515136171011412265[244] = 0.0;
   out_515136171011412265[245] = 0.0;
   out_515136171011412265[246] = 0.0;
   out_515136171011412265[247] = 1.0;
   out_515136171011412265[248] = 0.0;
   out_515136171011412265[249] = 0.0;
   out_515136171011412265[250] = 0.0;
   out_515136171011412265[251] = 0.0;
   out_515136171011412265[252] = 0.0;
   out_515136171011412265[253] = 0.0;
   out_515136171011412265[254] = 0.0;
   out_515136171011412265[255] = 0.0;
   out_515136171011412265[256] = 0.0;
   out_515136171011412265[257] = 0.0;
   out_515136171011412265[258] = 0.0;
   out_515136171011412265[259] = 0.0;
   out_515136171011412265[260] = 0.0;
   out_515136171011412265[261] = 0.0;
   out_515136171011412265[262] = 0.0;
   out_515136171011412265[263] = 0.0;
   out_515136171011412265[264] = 0.0;
   out_515136171011412265[265] = 0.0;
   out_515136171011412265[266] = 1.0;
   out_515136171011412265[267] = 0.0;
   out_515136171011412265[268] = 0.0;
   out_515136171011412265[269] = 0.0;
   out_515136171011412265[270] = 0.0;
   out_515136171011412265[271] = 0.0;
   out_515136171011412265[272] = 0.0;
   out_515136171011412265[273] = 0.0;
   out_515136171011412265[274] = 0.0;
   out_515136171011412265[275] = 0.0;
   out_515136171011412265[276] = 0.0;
   out_515136171011412265[277] = 0.0;
   out_515136171011412265[278] = 0.0;
   out_515136171011412265[279] = 0.0;
   out_515136171011412265[280] = 0.0;
   out_515136171011412265[281] = 0.0;
   out_515136171011412265[282] = 0.0;
   out_515136171011412265[283] = 0.0;
   out_515136171011412265[284] = 0.0;
   out_515136171011412265[285] = 1.0;
   out_515136171011412265[286] = 0.0;
   out_515136171011412265[287] = 0.0;
   out_515136171011412265[288] = 0.0;
   out_515136171011412265[289] = 0.0;
   out_515136171011412265[290] = 0.0;
   out_515136171011412265[291] = 0.0;
   out_515136171011412265[292] = 0.0;
   out_515136171011412265[293] = 0.0;
   out_515136171011412265[294] = 0.0;
   out_515136171011412265[295] = 0.0;
   out_515136171011412265[296] = 0.0;
   out_515136171011412265[297] = 0.0;
   out_515136171011412265[298] = 0.0;
   out_515136171011412265[299] = 0.0;
   out_515136171011412265[300] = 0.0;
   out_515136171011412265[301] = 0.0;
   out_515136171011412265[302] = 0.0;
   out_515136171011412265[303] = 0.0;
   out_515136171011412265[304] = 1.0;
   out_515136171011412265[305] = 0.0;
   out_515136171011412265[306] = 0.0;
   out_515136171011412265[307] = 0.0;
   out_515136171011412265[308] = 0.0;
   out_515136171011412265[309] = 0.0;
   out_515136171011412265[310] = 0.0;
   out_515136171011412265[311] = 0.0;
   out_515136171011412265[312] = 0.0;
   out_515136171011412265[313] = 0.0;
   out_515136171011412265[314] = 0.0;
   out_515136171011412265[315] = 0.0;
   out_515136171011412265[316] = 0.0;
   out_515136171011412265[317] = 0.0;
   out_515136171011412265[318] = 0.0;
   out_515136171011412265[319] = 0.0;
   out_515136171011412265[320] = 0.0;
   out_515136171011412265[321] = 0.0;
   out_515136171011412265[322] = 0.0;
   out_515136171011412265[323] = 1.0;
}
void f_fun(double *state, double dt, double *out_1656902809266479078) {
   out_1656902809266479078[0] = atan2((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), -(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]));
   out_1656902809266479078[1] = asin(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]));
   out_1656902809266479078[2] = atan2(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), -(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]));
   out_1656902809266479078[3] = dt*state[12] + state[3];
   out_1656902809266479078[4] = dt*state[13] + state[4];
   out_1656902809266479078[5] = dt*state[14] + state[5];
   out_1656902809266479078[6] = state[6];
   out_1656902809266479078[7] = state[7];
   out_1656902809266479078[8] = state[8];
   out_1656902809266479078[9] = state[9];
   out_1656902809266479078[10] = state[10];
   out_1656902809266479078[11] = state[11];
   out_1656902809266479078[12] = state[12];
   out_1656902809266479078[13] = state[13];
   out_1656902809266479078[14] = state[14];
   out_1656902809266479078[15] = state[15];
   out_1656902809266479078[16] = state[16];
   out_1656902809266479078[17] = state[17];
}
void F_fun(double *state, double dt, double *out_4521003238642649691) {
   out_4521003238642649691[0] = ((-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*cos(state[0])*cos(state[1]) - sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*cos(state[0])*cos(state[1]) - sin(dt*state[6])*sin(state[0])*cos(dt*state[7])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4521003238642649691[1] = ((-sin(dt*state[6])*sin(dt*state[8]) - sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*cos(state[1]) - (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*sin(state[1]) - sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(state[0]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*sin(state[1]) + (-sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) + sin(dt*state[8])*cos(dt*state[6]))*cos(state[1]) - sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(state[0]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4521003238642649691[2] = 0;
   out_4521003238642649691[3] = 0;
   out_4521003238642649691[4] = 0;
   out_4521003238642649691[5] = 0;
   out_4521003238642649691[6] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(dt*cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) - dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4521003238642649691[7] = (-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[6])*sin(dt*state[7])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[6])*sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) - dt*sin(dt*state[6])*sin(state[1])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + (-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))*(-dt*sin(dt*state[7])*cos(dt*state[6])*cos(state[0])*cos(state[1]) + dt*sin(dt*state[8])*sin(state[0])*cos(dt*state[6])*cos(dt*state[7])*cos(state[1]) - dt*sin(state[1])*cos(dt*state[6])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4521003238642649691[8] = ((dt*sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + dt*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (dt*sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]))*(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2)) + ((dt*sin(dt*state[6])*sin(dt*state[8]) + dt*sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (-dt*sin(dt*state[6])*cos(dt*state[8]) + dt*sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]))*(-(sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) + (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) - sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/(pow(-(sin(dt*state[6])*sin(dt*state[8]) + sin(dt*state[7])*cos(dt*state[6])*cos(dt*state[8]))*sin(state[1]) + (-sin(dt*state[6])*cos(dt*state[8]) + sin(dt*state[7])*sin(dt*state[8])*cos(dt*state[6]))*sin(state[0])*cos(state[1]) + cos(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2) + pow((sin(dt*state[6])*sin(dt*state[7])*sin(dt*state[8]) + cos(dt*state[6])*cos(dt*state[8]))*sin(state[0])*cos(state[1]) - (sin(dt*state[6])*sin(dt*state[7])*cos(dt*state[8]) - sin(dt*state[8])*cos(dt*state[6]))*sin(state[1]) + sin(dt*state[6])*cos(dt*state[7])*cos(state[0])*cos(state[1]), 2));
   out_4521003238642649691[9] = 0;
   out_4521003238642649691[10] = 0;
   out_4521003238642649691[11] = 0;
   out_4521003238642649691[12] = 0;
   out_4521003238642649691[13] = 0;
   out_4521003238642649691[14] = 0;
   out_4521003238642649691[15] = 0;
   out_4521003238642649691[16] = 0;
   out_4521003238642649691[17] = 0;
   out_4521003238642649691[18] = (-sin(dt*state[7])*sin(state[0])*cos(state[1]) - sin(dt*state[8])*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4521003238642649691[19] = (-sin(dt*state[7])*sin(state[1])*cos(state[0]) + sin(dt*state[8])*sin(state[0])*sin(state[1])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4521003238642649691[20] = 0;
   out_4521003238642649691[21] = 0;
   out_4521003238642649691[22] = 0;
   out_4521003238642649691[23] = 0;
   out_4521003238642649691[24] = 0;
   out_4521003238642649691[25] = (dt*sin(dt*state[7])*sin(dt*state[8])*sin(state[0])*cos(state[1]) - dt*sin(dt*state[7])*sin(state[1])*cos(dt*state[8]) + dt*cos(dt*state[7])*cos(state[0])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4521003238642649691[26] = (-dt*sin(dt*state[8])*sin(state[1])*cos(dt*state[7]) - dt*sin(state[0])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/sqrt(1 - pow(sin(dt*state[7])*cos(state[0])*cos(state[1]) - sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1]) + sin(state[1])*cos(dt*state[7])*cos(dt*state[8]), 2));
   out_4521003238642649691[27] = 0;
   out_4521003238642649691[28] = 0;
   out_4521003238642649691[29] = 0;
   out_4521003238642649691[30] = 0;
   out_4521003238642649691[31] = 0;
   out_4521003238642649691[32] = 0;
   out_4521003238642649691[33] = 0;
   out_4521003238642649691[34] = 0;
   out_4521003238642649691[35] = 0;
   out_4521003238642649691[36] = ((sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4521003238642649691[37] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-sin(dt*state[7])*sin(state[2])*cos(state[0])*cos(state[1]) + sin(dt*state[8])*sin(state[0])*sin(state[2])*cos(dt*state[7])*cos(state[1]) - sin(state[1])*sin(state[2])*cos(dt*state[7])*cos(dt*state[8]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(-sin(dt*state[7])*cos(state[0])*cos(state[1])*cos(state[2]) + sin(dt*state[8])*sin(state[0])*cos(dt*state[7])*cos(state[1])*cos(state[2]) - sin(state[1])*cos(dt*state[7])*cos(dt*state[8])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4521003238642649691[38] = ((-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (-sin(state[0])*sin(state[1])*sin(state[2]) - cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4521003238642649691[39] = 0;
   out_4521003238642649691[40] = 0;
   out_4521003238642649691[41] = 0;
   out_4521003238642649691[42] = 0;
   out_4521003238642649691[43] = (-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))*(dt*(sin(state[0])*cos(state[2]) - sin(state[1])*sin(state[2])*cos(state[0]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*sin(state[2])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + ((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))*(dt*(-sin(state[0])*sin(state[2]) - sin(state[1])*cos(state[0])*cos(state[2]))*cos(dt*state[7]) - dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[7])*sin(dt*state[8]) - dt*sin(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4521003238642649691[44] = (dt*(sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*sin(state[2])*cos(dt*state[7])*cos(state[1]))*(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2)) + (dt*(sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*cos(dt*state[7])*cos(dt*state[8]) - dt*sin(dt*state[8])*cos(dt*state[7])*cos(state[1])*cos(state[2]))*((-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) - (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) - sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]))/(pow(-(sin(state[0])*sin(state[2]) + sin(state[1])*cos(state[0])*cos(state[2]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*cos(state[2]) - sin(state[2])*cos(state[0]))*sin(dt*state[8])*cos(dt*state[7]) + cos(dt*state[7])*cos(dt*state[8])*cos(state[1])*cos(state[2]), 2) + pow(-(-sin(state[0])*cos(state[2]) + sin(state[1])*sin(state[2])*cos(state[0]))*sin(dt*state[7]) + (sin(state[0])*sin(state[1])*sin(state[2]) + cos(state[0])*cos(state[2]))*sin(dt*state[8])*cos(dt*state[7]) + sin(state[2])*cos(dt*state[7])*cos(dt*state[8])*cos(state[1]), 2));
   out_4521003238642649691[45] = 0;
   out_4521003238642649691[46] = 0;
   out_4521003238642649691[47] = 0;
   out_4521003238642649691[48] = 0;
   out_4521003238642649691[49] = 0;
   out_4521003238642649691[50] = 0;
   out_4521003238642649691[51] = 0;
   out_4521003238642649691[52] = 0;
   out_4521003238642649691[53] = 0;
   out_4521003238642649691[54] = 0;
   out_4521003238642649691[55] = 0;
   out_4521003238642649691[56] = 0;
   out_4521003238642649691[57] = 1;
   out_4521003238642649691[58] = 0;
   out_4521003238642649691[59] = 0;
   out_4521003238642649691[60] = 0;
   out_4521003238642649691[61] = 0;
   out_4521003238642649691[62] = 0;
   out_4521003238642649691[63] = 0;
   out_4521003238642649691[64] = 0;
   out_4521003238642649691[65] = 0;
   out_4521003238642649691[66] = dt;
   out_4521003238642649691[67] = 0;
   out_4521003238642649691[68] = 0;
   out_4521003238642649691[69] = 0;
   out_4521003238642649691[70] = 0;
   out_4521003238642649691[71] = 0;
   out_4521003238642649691[72] = 0;
   out_4521003238642649691[73] = 0;
   out_4521003238642649691[74] = 0;
   out_4521003238642649691[75] = 0;
   out_4521003238642649691[76] = 1;
   out_4521003238642649691[77] = 0;
   out_4521003238642649691[78] = 0;
   out_4521003238642649691[79] = 0;
   out_4521003238642649691[80] = 0;
   out_4521003238642649691[81] = 0;
   out_4521003238642649691[82] = 0;
   out_4521003238642649691[83] = 0;
   out_4521003238642649691[84] = 0;
   out_4521003238642649691[85] = dt;
   out_4521003238642649691[86] = 0;
   out_4521003238642649691[87] = 0;
   out_4521003238642649691[88] = 0;
   out_4521003238642649691[89] = 0;
   out_4521003238642649691[90] = 0;
   out_4521003238642649691[91] = 0;
   out_4521003238642649691[92] = 0;
   out_4521003238642649691[93] = 0;
   out_4521003238642649691[94] = 0;
   out_4521003238642649691[95] = 1;
   out_4521003238642649691[96] = 0;
   out_4521003238642649691[97] = 0;
   out_4521003238642649691[98] = 0;
   out_4521003238642649691[99] = 0;
   out_4521003238642649691[100] = 0;
   out_4521003238642649691[101] = 0;
   out_4521003238642649691[102] = 0;
   out_4521003238642649691[103] = 0;
   out_4521003238642649691[104] = dt;
   out_4521003238642649691[105] = 0;
   out_4521003238642649691[106] = 0;
   out_4521003238642649691[107] = 0;
   out_4521003238642649691[108] = 0;
   out_4521003238642649691[109] = 0;
   out_4521003238642649691[110] = 0;
   out_4521003238642649691[111] = 0;
   out_4521003238642649691[112] = 0;
   out_4521003238642649691[113] = 0;
   out_4521003238642649691[114] = 1;
   out_4521003238642649691[115] = 0;
   out_4521003238642649691[116] = 0;
   out_4521003238642649691[117] = 0;
   out_4521003238642649691[118] = 0;
   out_4521003238642649691[119] = 0;
   out_4521003238642649691[120] = 0;
   out_4521003238642649691[121] = 0;
   out_4521003238642649691[122] = 0;
   out_4521003238642649691[123] = 0;
   out_4521003238642649691[124] = 0;
   out_4521003238642649691[125] = 0;
   out_4521003238642649691[126] = 0;
   out_4521003238642649691[127] = 0;
   out_4521003238642649691[128] = 0;
   out_4521003238642649691[129] = 0;
   out_4521003238642649691[130] = 0;
   out_4521003238642649691[131] = 0;
   out_4521003238642649691[132] = 0;
   out_4521003238642649691[133] = 1;
   out_4521003238642649691[134] = 0;
   out_4521003238642649691[135] = 0;
   out_4521003238642649691[136] = 0;
   out_4521003238642649691[137] = 0;
   out_4521003238642649691[138] = 0;
   out_4521003238642649691[139] = 0;
   out_4521003238642649691[140] = 0;
   out_4521003238642649691[141] = 0;
   out_4521003238642649691[142] = 0;
   out_4521003238642649691[143] = 0;
   out_4521003238642649691[144] = 0;
   out_4521003238642649691[145] = 0;
   out_4521003238642649691[146] = 0;
   out_4521003238642649691[147] = 0;
   out_4521003238642649691[148] = 0;
   out_4521003238642649691[149] = 0;
   out_4521003238642649691[150] = 0;
   out_4521003238642649691[151] = 0;
   out_4521003238642649691[152] = 1;
   out_4521003238642649691[153] = 0;
   out_4521003238642649691[154] = 0;
   out_4521003238642649691[155] = 0;
   out_4521003238642649691[156] = 0;
   out_4521003238642649691[157] = 0;
   out_4521003238642649691[158] = 0;
   out_4521003238642649691[159] = 0;
   out_4521003238642649691[160] = 0;
   out_4521003238642649691[161] = 0;
   out_4521003238642649691[162] = 0;
   out_4521003238642649691[163] = 0;
   out_4521003238642649691[164] = 0;
   out_4521003238642649691[165] = 0;
   out_4521003238642649691[166] = 0;
   out_4521003238642649691[167] = 0;
   out_4521003238642649691[168] = 0;
   out_4521003238642649691[169] = 0;
   out_4521003238642649691[170] = 0;
   out_4521003238642649691[171] = 1;
   out_4521003238642649691[172] = 0;
   out_4521003238642649691[173] = 0;
   out_4521003238642649691[174] = 0;
   out_4521003238642649691[175] = 0;
   out_4521003238642649691[176] = 0;
   out_4521003238642649691[177] = 0;
   out_4521003238642649691[178] = 0;
   out_4521003238642649691[179] = 0;
   out_4521003238642649691[180] = 0;
   out_4521003238642649691[181] = 0;
   out_4521003238642649691[182] = 0;
   out_4521003238642649691[183] = 0;
   out_4521003238642649691[184] = 0;
   out_4521003238642649691[185] = 0;
   out_4521003238642649691[186] = 0;
   out_4521003238642649691[187] = 0;
   out_4521003238642649691[188] = 0;
   out_4521003238642649691[189] = 0;
   out_4521003238642649691[190] = 1;
   out_4521003238642649691[191] = 0;
   out_4521003238642649691[192] = 0;
   out_4521003238642649691[193] = 0;
   out_4521003238642649691[194] = 0;
   out_4521003238642649691[195] = 0;
   out_4521003238642649691[196] = 0;
   out_4521003238642649691[197] = 0;
   out_4521003238642649691[198] = 0;
   out_4521003238642649691[199] = 0;
   out_4521003238642649691[200] = 0;
   out_4521003238642649691[201] = 0;
   out_4521003238642649691[202] = 0;
   out_4521003238642649691[203] = 0;
   out_4521003238642649691[204] = 0;
   out_4521003238642649691[205] = 0;
   out_4521003238642649691[206] = 0;
   out_4521003238642649691[207] = 0;
   out_4521003238642649691[208] = 0;
   out_4521003238642649691[209] = 1;
   out_4521003238642649691[210] = 0;
   out_4521003238642649691[211] = 0;
   out_4521003238642649691[212] = 0;
   out_4521003238642649691[213] = 0;
   out_4521003238642649691[214] = 0;
   out_4521003238642649691[215] = 0;
   out_4521003238642649691[216] = 0;
   out_4521003238642649691[217] = 0;
   out_4521003238642649691[218] = 0;
   out_4521003238642649691[219] = 0;
   out_4521003238642649691[220] = 0;
   out_4521003238642649691[221] = 0;
   out_4521003238642649691[222] = 0;
   out_4521003238642649691[223] = 0;
   out_4521003238642649691[224] = 0;
   out_4521003238642649691[225] = 0;
   out_4521003238642649691[226] = 0;
   out_4521003238642649691[227] = 0;
   out_4521003238642649691[228] = 1;
   out_4521003238642649691[229] = 0;
   out_4521003238642649691[230] = 0;
   out_4521003238642649691[231] = 0;
   out_4521003238642649691[232] = 0;
   out_4521003238642649691[233] = 0;
   out_4521003238642649691[234] = 0;
   out_4521003238642649691[235] = 0;
   out_4521003238642649691[236] = 0;
   out_4521003238642649691[237] = 0;
   out_4521003238642649691[238] = 0;
   out_4521003238642649691[239] = 0;
   out_4521003238642649691[240] = 0;
   out_4521003238642649691[241] = 0;
   out_4521003238642649691[242] = 0;
   out_4521003238642649691[243] = 0;
   out_4521003238642649691[244] = 0;
   out_4521003238642649691[245] = 0;
   out_4521003238642649691[246] = 0;
   out_4521003238642649691[247] = 1;
   out_4521003238642649691[248] = 0;
   out_4521003238642649691[249] = 0;
   out_4521003238642649691[250] = 0;
   out_4521003238642649691[251] = 0;
   out_4521003238642649691[252] = 0;
   out_4521003238642649691[253] = 0;
   out_4521003238642649691[254] = 0;
   out_4521003238642649691[255] = 0;
   out_4521003238642649691[256] = 0;
   out_4521003238642649691[257] = 0;
   out_4521003238642649691[258] = 0;
   out_4521003238642649691[259] = 0;
   out_4521003238642649691[260] = 0;
   out_4521003238642649691[261] = 0;
   out_4521003238642649691[262] = 0;
   out_4521003238642649691[263] = 0;
   out_4521003238642649691[264] = 0;
   out_4521003238642649691[265] = 0;
   out_4521003238642649691[266] = 1;
   out_4521003238642649691[267] = 0;
   out_4521003238642649691[268] = 0;
   out_4521003238642649691[269] = 0;
   out_4521003238642649691[270] = 0;
   out_4521003238642649691[271] = 0;
   out_4521003238642649691[272] = 0;
   out_4521003238642649691[273] = 0;
   out_4521003238642649691[274] = 0;
   out_4521003238642649691[275] = 0;
   out_4521003238642649691[276] = 0;
   out_4521003238642649691[277] = 0;
   out_4521003238642649691[278] = 0;
   out_4521003238642649691[279] = 0;
   out_4521003238642649691[280] = 0;
   out_4521003238642649691[281] = 0;
   out_4521003238642649691[282] = 0;
   out_4521003238642649691[283] = 0;
   out_4521003238642649691[284] = 0;
   out_4521003238642649691[285] = 1;
   out_4521003238642649691[286] = 0;
   out_4521003238642649691[287] = 0;
   out_4521003238642649691[288] = 0;
   out_4521003238642649691[289] = 0;
   out_4521003238642649691[290] = 0;
   out_4521003238642649691[291] = 0;
   out_4521003238642649691[292] = 0;
   out_4521003238642649691[293] = 0;
   out_4521003238642649691[294] = 0;
   out_4521003238642649691[295] = 0;
   out_4521003238642649691[296] = 0;
   out_4521003238642649691[297] = 0;
   out_4521003238642649691[298] = 0;
   out_4521003238642649691[299] = 0;
   out_4521003238642649691[300] = 0;
   out_4521003238642649691[301] = 0;
   out_4521003238642649691[302] = 0;
   out_4521003238642649691[303] = 0;
   out_4521003238642649691[304] = 1;
   out_4521003238642649691[305] = 0;
   out_4521003238642649691[306] = 0;
   out_4521003238642649691[307] = 0;
   out_4521003238642649691[308] = 0;
   out_4521003238642649691[309] = 0;
   out_4521003238642649691[310] = 0;
   out_4521003238642649691[311] = 0;
   out_4521003238642649691[312] = 0;
   out_4521003238642649691[313] = 0;
   out_4521003238642649691[314] = 0;
   out_4521003238642649691[315] = 0;
   out_4521003238642649691[316] = 0;
   out_4521003238642649691[317] = 0;
   out_4521003238642649691[318] = 0;
   out_4521003238642649691[319] = 0;
   out_4521003238642649691[320] = 0;
   out_4521003238642649691[321] = 0;
   out_4521003238642649691[322] = 0;
   out_4521003238642649691[323] = 1;
}
void h_4(double *state, double *unused, double *out_6145288799047026529) {
   out_6145288799047026529[0] = state[6] + state[9];
   out_6145288799047026529[1] = state[7] + state[10];
   out_6145288799047026529[2] = state[8] + state[11];
}
void H_4(double *state, double *unused, double *out_5332945635740546456) {
   out_5332945635740546456[0] = 0;
   out_5332945635740546456[1] = 0;
   out_5332945635740546456[2] = 0;
   out_5332945635740546456[3] = 0;
   out_5332945635740546456[4] = 0;
   out_5332945635740546456[5] = 0;
   out_5332945635740546456[6] = 1;
   out_5332945635740546456[7] = 0;
   out_5332945635740546456[8] = 0;
   out_5332945635740546456[9] = 1;
   out_5332945635740546456[10] = 0;
   out_5332945635740546456[11] = 0;
   out_5332945635740546456[12] = 0;
   out_5332945635740546456[13] = 0;
   out_5332945635740546456[14] = 0;
   out_5332945635740546456[15] = 0;
   out_5332945635740546456[16] = 0;
   out_5332945635740546456[17] = 0;
   out_5332945635740546456[18] = 0;
   out_5332945635740546456[19] = 0;
   out_5332945635740546456[20] = 0;
   out_5332945635740546456[21] = 0;
   out_5332945635740546456[22] = 0;
   out_5332945635740546456[23] = 0;
   out_5332945635740546456[24] = 0;
   out_5332945635740546456[25] = 1;
   out_5332945635740546456[26] = 0;
   out_5332945635740546456[27] = 0;
   out_5332945635740546456[28] = 1;
   out_5332945635740546456[29] = 0;
   out_5332945635740546456[30] = 0;
   out_5332945635740546456[31] = 0;
   out_5332945635740546456[32] = 0;
   out_5332945635740546456[33] = 0;
   out_5332945635740546456[34] = 0;
   out_5332945635740546456[35] = 0;
   out_5332945635740546456[36] = 0;
   out_5332945635740546456[37] = 0;
   out_5332945635740546456[38] = 0;
   out_5332945635740546456[39] = 0;
   out_5332945635740546456[40] = 0;
   out_5332945635740546456[41] = 0;
   out_5332945635740546456[42] = 0;
   out_5332945635740546456[43] = 0;
   out_5332945635740546456[44] = 1;
   out_5332945635740546456[45] = 0;
   out_5332945635740546456[46] = 0;
   out_5332945635740546456[47] = 1;
   out_5332945635740546456[48] = 0;
   out_5332945635740546456[49] = 0;
   out_5332945635740546456[50] = 0;
   out_5332945635740546456[51] = 0;
   out_5332945635740546456[52] = 0;
   out_5332945635740546456[53] = 0;
}
void h_10(double *state, double *unused, double *out_569963962585618949) {
   out_569963962585618949[0] = 9.8100000000000005*sin(state[1]) - state[4]*state[8] + state[5]*state[7] + state[12] + state[15];
   out_569963962585618949[1] = -9.8100000000000005*sin(state[0])*cos(state[1]) + state[3]*state[8] - state[5]*state[6] + state[13] + state[16];
   out_569963962585618949[2] = -9.8100000000000005*cos(state[0])*cos(state[1]) - state[3]*state[7] + state[4]*state[6] + state[14] + state[17];
}
void H_10(double *state, double *unused, double *out_7139448533129631908) {
   out_7139448533129631908[0] = 0;
   out_7139448533129631908[1] = 9.8100000000000005*cos(state[1]);
   out_7139448533129631908[2] = 0;
   out_7139448533129631908[3] = 0;
   out_7139448533129631908[4] = -state[8];
   out_7139448533129631908[5] = state[7];
   out_7139448533129631908[6] = 0;
   out_7139448533129631908[7] = state[5];
   out_7139448533129631908[8] = -state[4];
   out_7139448533129631908[9] = 0;
   out_7139448533129631908[10] = 0;
   out_7139448533129631908[11] = 0;
   out_7139448533129631908[12] = 1;
   out_7139448533129631908[13] = 0;
   out_7139448533129631908[14] = 0;
   out_7139448533129631908[15] = 1;
   out_7139448533129631908[16] = 0;
   out_7139448533129631908[17] = 0;
   out_7139448533129631908[18] = -9.8100000000000005*cos(state[0])*cos(state[1]);
   out_7139448533129631908[19] = 9.8100000000000005*sin(state[0])*sin(state[1]);
   out_7139448533129631908[20] = 0;
   out_7139448533129631908[21] = state[8];
   out_7139448533129631908[22] = 0;
   out_7139448533129631908[23] = -state[6];
   out_7139448533129631908[24] = -state[5];
   out_7139448533129631908[25] = 0;
   out_7139448533129631908[26] = state[3];
   out_7139448533129631908[27] = 0;
   out_7139448533129631908[28] = 0;
   out_7139448533129631908[29] = 0;
   out_7139448533129631908[30] = 0;
   out_7139448533129631908[31] = 1;
   out_7139448533129631908[32] = 0;
   out_7139448533129631908[33] = 0;
   out_7139448533129631908[34] = 1;
   out_7139448533129631908[35] = 0;
   out_7139448533129631908[36] = 9.8100000000000005*sin(state[0])*cos(state[1]);
   out_7139448533129631908[37] = 9.8100000000000005*sin(state[1])*cos(state[0]);
   out_7139448533129631908[38] = 0;
   out_7139448533129631908[39] = -state[7];
   out_7139448533129631908[40] = state[6];
   out_7139448533129631908[41] = 0;
   out_7139448533129631908[42] = state[4];
   out_7139448533129631908[43] = -state[3];
   out_7139448533129631908[44] = 0;
   out_7139448533129631908[45] = 0;
   out_7139448533129631908[46] = 0;
   out_7139448533129631908[47] = 0;
   out_7139448533129631908[48] = 0;
   out_7139448533129631908[49] = 0;
   out_7139448533129631908[50] = 1;
   out_7139448533129631908[51] = 0;
   out_7139448533129631908[52] = 0;
   out_7139448533129631908[53] = 1;
}
void h_13(double *state, double *unused, double *out_7745490486830301841) {
   out_7745490486830301841[0] = state[3];
   out_7745490486830301841[1] = state[4];
   out_7745490486830301841[2] = state[5];
}
void H_13(double *state, double *unused, double *out_2277685572576154473) {
   out_2277685572576154473[0] = 0;
   out_2277685572576154473[1] = 0;
   out_2277685572576154473[2] = 0;
   out_2277685572576154473[3] = 1;
   out_2277685572576154473[4] = 0;
   out_2277685572576154473[5] = 0;
   out_2277685572576154473[6] = 0;
   out_2277685572576154473[7] = 0;
   out_2277685572576154473[8] = 0;
   out_2277685572576154473[9] = 0;
   out_2277685572576154473[10] = 0;
   out_2277685572576154473[11] = 0;
   out_2277685572576154473[12] = 0;
   out_2277685572576154473[13] = 0;
   out_2277685572576154473[14] = 0;
   out_2277685572576154473[15] = 0;
   out_2277685572576154473[16] = 0;
   out_2277685572576154473[17] = 0;
   out_2277685572576154473[18] = 0;
   out_2277685572576154473[19] = 0;
   out_2277685572576154473[20] = 0;
   out_2277685572576154473[21] = 0;
   out_2277685572576154473[22] = 1;
   out_2277685572576154473[23] = 0;
   out_2277685572576154473[24] = 0;
   out_2277685572576154473[25] = 0;
   out_2277685572576154473[26] = 0;
   out_2277685572576154473[27] = 0;
   out_2277685572576154473[28] = 0;
   out_2277685572576154473[29] = 0;
   out_2277685572576154473[30] = 0;
   out_2277685572576154473[31] = 0;
   out_2277685572576154473[32] = 0;
   out_2277685572576154473[33] = 0;
   out_2277685572576154473[34] = 0;
   out_2277685572576154473[35] = 0;
   out_2277685572576154473[36] = 0;
   out_2277685572576154473[37] = 0;
   out_2277685572576154473[38] = 0;
   out_2277685572576154473[39] = 0;
   out_2277685572576154473[40] = 0;
   out_2277685572576154473[41] = 1;
   out_2277685572576154473[42] = 0;
   out_2277685572576154473[43] = 0;
   out_2277685572576154473[44] = 0;
   out_2277685572576154473[45] = 0;
   out_2277685572576154473[46] = 0;
   out_2277685572576154473[47] = 0;
   out_2277685572576154473[48] = 0;
   out_2277685572576154473[49] = 0;
   out_2277685572576154473[50] = 0;
   out_2277685572576154473[51] = 0;
   out_2277685572576154473[52] = 0;
   out_2277685572576154473[53] = 0;
}
void h_14(double *state, double *unused, double *out_7326154253764583640) {
   out_7326154253764583640[0] = state[6];
   out_7326154253764583640[1] = state[7];
   out_7326154253764583640[2] = state[8];
}
void H_14(double *state, double *unused, double *out_1369704779401061927) {
   out_1369704779401061927[0] = 0;
   out_1369704779401061927[1] = 0;
   out_1369704779401061927[2] = 0;
   out_1369704779401061927[3] = 0;
   out_1369704779401061927[4] = 0;
   out_1369704779401061927[5] = 0;
   out_1369704779401061927[6] = 1;
   out_1369704779401061927[7] = 0;
   out_1369704779401061927[8] = 0;
   out_1369704779401061927[9] = 0;
   out_1369704779401061927[10] = 0;
   out_1369704779401061927[11] = 0;
   out_1369704779401061927[12] = 0;
   out_1369704779401061927[13] = 0;
   out_1369704779401061927[14] = 0;
   out_1369704779401061927[15] = 0;
   out_1369704779401061927[16] = 0;
   out_1369704779401061927[17] = 0;
   out_1369704779401061927[18] = 0;
   out_1369704779401061927[19] = 0;
   out_1369704779401061927[20] = 0;
   out_1369704779401061927[21] = 0;
   out_1369704779401061927[22] = 0;
   out_1369704779401061927[23] = 0;
   out_1369704779401061927[24] = 0;
   out_1369704779401061927[25] = 1;
   out_1369704779401061927[26] = 0;
   out_1369704779401061927[27] = 0;
   out_1369704779401061927[28] = 0;
   out_1369704779401061927[29] = 0;
   out_1369704779401061927[30] = 0;
   out_1369704779401061927[31] = 0;
   out_1369704779401061927[32] = 0;
   out_1369704779401061927[33] = 0;
   out_1369704779401061927[34] = 0;
   out_1369704779401061927[35] = 0;
   out_1369704779401061927[36] = 0;
   out_1369704779401061927[37] = 0;
   out_1369704779401061927[38] = 0;
   out_1369704779401061927[39] = 0;
   out_1369704779401061927[40] = 0;
   out_1369704779401061927[41] = 0;
   out_1369704779401061927[42] = 0;
   out_1369704779401061927[43] = 0;
   out_1369704779401061927[44] = 1;
   out_1369704779401061927[45] = 0;
   out_1369704779401061927[46] = 0;
   out_1369704779401061927[47] = 0;
   out_1369704779401061927[48] = 0;
   out_1369704779401061927[49] = 0;
   out_1369704779401061927[50] = 0;
   out_1369704779401061927[51] = 0;
   out_1369704779401061927[52] = 0;
   out_1369704779401061927[53] = 0;
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
void pose_err_fun(double *nom_x, double *delta_x, double *out_2016014972386344778) {
  err_fun(nom_x, delta_x, out_2016014972386344778);
}
void pose_inv_err_fun(double *nom_x, double *true_x, double *out_7065715320158728118) {
  inv_err_fun(nom_x, true_x, out_7065715320158728118);
}
void pose_H_mod_fun(double *state, double *out_515136171011412265) {
  H_mod_fun(state, out_515136171011412265);
}
void pose_f_fun(double *state, double dt, double *out_1656902809266479078) {
  f_fun(state,  dt, out_1656902809266479078);
}
void pose_F_fun(double *state, double dt, double *out_4521003238642649691) {
  F_fun(state,  dt, out_4521003238642649691);
}
void pose_h_4(double *state, double *unused, double *out_6145288799047026529) {
  h_4(state, unused, out_6145288799047026529);
}
void pose_H_4(double *state, double *unused, double *out_5332945635740546456) {
  H_4(state, unused, out_5332945635740546456);
}
void pose_h_10(double *state, double *unused, double *out_569963962585618949) {
  h_10(state, unused, out_569963962585618949);
}
void pose_H_10(double *state, double *unused, double *out_7139448533129631908) {
  H_10(state, unused, out_7139448533129631908);
}
void pose_h_13(double *state, double *unused, double *out_7745490486830301841) {
  h_13(state, unused, out_7745490486830301841);
}
void pose_H_13(double *state, double *unused, double *out_2277685572576154473) {
  H_13(state, unused, out_2277685572576154473);
}
void pose_h_14(double *state, double *unused, double *out_7326154253764583640) {
  h_14(state, unused, out_7326154253764583640);
}
void pose_H_14(double *state, double *unused, double *out_1369704779401061927) {
  H_14(state, unused, out_1369704779401061927);
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
