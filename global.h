/* define some global variables */

#ifndef _GLOBAL_H
#define _GLOBAL_H


#define _GSI(x) 24*x
#define T_global 32
#define LX_global 32
#define LY_global 32
#define LZ_global 32

const int T = 32;
const int LX = 32;
const int LY = 32;
const int LZ = 32;

const int kernel_n=3;
const int kernel_n_geom=5;

const int idx_comb[6][2] = {
  {0,1},
  {0,2},
  {0,3},
  {1,2},
  {1,3},
  {2,3} };

#endif