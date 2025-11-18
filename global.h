/* define some global variables */

#ifndef _GLOBAL_H
#define _GLOBAL_H

//#include <mpi.h>

#define _GSI(x) 24*x
#define T_global 32
#define LX_global 32
#define LY_global 32
#define LZ_global 32


#define NPROCT 1
#define NPROCX 1
#define NPROCY 1
#define NPROCZ 1 

const int T = T_global/NPROCT;
const int LX = LX_global/NPROCX;
const int LY = LX_global/NPROCY;
const int LZ = LX_global/NPROCZ;

const int kernel_n=3;
const int kernel_n_geom=3;

const int idx_comb[6][2] = {
  {0,1},
  {0,2},
  {0,3},
  {1,2},
  {1,3},
  {2,3} };

// MPI
//MPI_Comm g_cart_grid;
//int g_cart_id;
int g_proc_coords[4]={0};
#endif