/* GPU kernels of 2p2
    JingJing Li, 2025
*/
#include <iostream>
//#include "cvc_complex.h"
//#include <cmath>
#include <stdio.h> 
#include <stdlib.h>
//#include "cvc_linalg.h"
#include <vector>
#include "global.h"
//#include "kernels.h"

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define PI_UNROLL 1

typedef struct {
  double re, im;
} complex;

__device__ __constant__ int gamma_permutation[16][24] = {
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16}
};
__device__ __constant__ int gamma_sign[16][24] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1}
};

__device__ __constant__ int idx_comb_d[6][2] ={
    {0,1},
    {0,2},
    {0,3},
    {1,2},
    {1,3},
    {2,3}
  };

#define _co_eq_fv_dag_ti_fv(c,s,t) {\
  (c)->re = \
    (s)[ 0]*(t)[ 0] + (s)[ 1]*(t)[ 1] +\
    (s)[ 2]*(t)[ 2] + (s)[ 3]*(t)[ 3] +\
    (s)[ 4]*(t)[ 4] + (s)[ 5]*(t)[ 5] +\
    (s)[ 6]*(t)[ 6] + (s)[ 7]*(t)[ 7] +\
    (s)[ 8]*(t)[ 8] + (s)[ 9]*(t)[ 9] +\
    (s)[10]*(t)[10] + (s)[11]*(t)[11] +\
    (s)[12]*(t)[12] + (s)[13]*(t)[13] +\
    (s)[14]*(t)[14] + (s)[15]*(t)[15] +\
    (s)[16]*(t)[16] + (s)[17]*(t)[17] +\
    (s)[18]*(t)[18] + (s)[19]*(t)[19] +\
    (s)[20]*(t)[20] + (s)[21]*(t)[21] +\
    (s)[22]*(t)[22] + (s)[23]*(t)[23];\
  (c)->im =\
    (s)[ 0]*(t)[ 1] - (s)[ 1]*(t)[ 0] +\
    (s)[ 2]*(t)[ 3] - (s)[ 3]*(t)[ 2] +\
    (s)[ 4]*(t)[ 5] - (s)[ 5]*(t)[ 4] +\
    (s)[ 6]*(t)[ 7] - (s)[ 7]*(t)[ 6] +\
    (s)[ 8]*(t)[ 9] - (s)[ 9]*(t)[ 8] +\
    (s)[10]*(t)[11] - (s)[11]*(t)[10] +\
    (s)[12]*(t)[13] - (s)[13]*(t)[12] +\
    (s)[14]*(t)[15] - (s)[15]*(t)[14] +\
    (s)[16]*(t)[17] - (s)[17]*(t)[16] +\
    (s)[18]*(t)[19] - (s)[19]*(t)[18] +\
    (s)[20]*(t)[21] - (s)[21]*(t)[20] +\
    (s)[22]*(t)[23] - (s)[23]*(t)[22];}


__device__ inline void _fv_eq_gamma_ti_fv(double* out, int gamma_index, const double* in) {
  for (int i = 0; i < 24; ++i) {
    out[i] = in[gamma_permutation[gamma_index][i]] * gamma_sign[gamma_index][i];
  }
}
__device__ inline void _fv_ti_eq_g5(double* in_out) {
  for (int i = 12; i < 24; ++i) {
    in_out[i] *= -1;
  }
}


__device__ inline static int get_global_Lmax()
{
    int Lmax = T_global;
    Lmax = (LX_global < Lmax) ? Lmax : LX_global;
    Lmax = (LY_global < Lmax) ? Lmax : LY_global;
    Lmax = (LZ_global < Lmax) ? Lmax : LZ_global;
    return Lmax;
}

__device__ inline static int get_local_Lmax()
{
    int Lmax = T;
    Lmax = (LX < Lmax) ? Lmax : LX;
    Lmax = (LY < Lmax) ? Lmax : LY;
    Lmax = (LZ < Lmax) ? Lmax : LZ;
    return Lmax;
}

__device__ inline static int prop_idx(int iflavour, int ia, int x, int ib) {
    return iflavour * 12 * 24 * LX_global * LY_global * LZ_global * T_global
         + ia * 24 * LX_global * LY_global * LZ_global * T_global
         + x * 24 + ib;
}

__device__ inline static void site_map_zerohalf (int xv[4], int const x[4] )
{
  xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
  xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
  xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
  xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

  return;
}


/* set a length len vector v to zero */
__device__ void set_zero(double *v, int len) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i=tid; i<len; i+=blockDim.x*gridDim.x){
        v[tid] = 0;
    }
}


//using namespace cvc;
//using namespace std;

__global__ void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned VOLUME){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
/*     __shared__ double u[12 * 24];
    __shared__ double d[12 * 24]; */

    // unroll over lattice sites by 2
    for (int ix = tid; ix < VOLUME; ix += gridDim.x * blockDim.x) {
        /* load 12 x 12 (x2 complex) d and u from fwd_y*/
        double u[12 * 24];
        double d[12 * 24];
        for (int ia = threadIdx.x; ia < 12; ia+=blockDim.x)
        for (int ib = threadIdx.y; ib < 24; ib+=blockDim.y) {
            u[ia * 24 + ib] = fwd_y[prop_idx(iflavor, ia, ix, ib)];
        }
        for (int ia = 0; ia < 12; ia++)
        for (int ib = 0; ib < 24; ib++){
            d[ia * 24 + ib] = fwd_y[prop_idx(1-iflavor, ia, ix, ib)];
        }

        /* loop over mu and nu */
        // use half the threads to compute pi at ix
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++) {
            double gu[12 * 24];
            double dot_prod[12 * 24];

            /* apply gammas: gu = g_5 g_mu u */
            for (int ia=0; ia<12; ia++) {
                _fv_eq_gamma_ti_fv(gu + ia * 24, mu, u + ia * 24);
                _fv_ti_eq_g5(gu + ia * 24);
            }

            /* compute <d, u>, dot_prod = <d, gmu> */
            for (int ia=0; ia<12; ia++ ) 
            for (int ib=0; ib<12; ib++ ) {
                complex w;
                _co_eq_fv_dag_ti_fv(&w, d + ib * 24, gu + ia * 24);
                dot_prod[ia * 24 + 2*ib] = w.re;
                dot_prod[ia * 24 + 2*ib + 1] = w.im;
            }

            /* apply gammas: gu = g_nu g_5 dot_prod */
            for (int ia=0; ia<12; ia++) {
                _fv_ti_eq_g5(dot_prod + ia * 24);
                _fv_eq_gamma_ti_fv(gu + ia * 24, nu, dot_prod + ia * 24);
            }

            /* take trace over ia: pi[x][mu][nu] = Tr[gu] */
            double trace = 0.0;
            for (int ia=0; ia<12; ia++) {trace += gu[ia * 24 + 2 * ia]; } // only real part
            Pi[ix*16+mu*4+nu] = trace;
        }
    }
}


/* warp reduction of an array, result stored in thread 0 */
__inline__ __device__ void reduction_warp(double *val){
    // loop over shuffle offsets (WARP_SIZE/2, WARP_SIZE/4, ..., 2, 1)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
        *val += __shfl_down_sync(FULL_MASK, *val, offset);
    }
}


/* block reduction, result stored in thread 0 */
__inline__ __device__ void reduction_block(double *val){
    static __shared__ double shared[WARP_SIZE]; 
    int lane = threadIdx.x % WARP_SIZE; // lane id in the warp
    int wid = threadIdx.x / WARP_SIZE; // warp id

    // each warp performs partial reduction
    reduction_warp(val);

    // write reduced value to shared memory
    if (lane == 0){
        shared[wid] = *val;
    }
    __syncthreads();

    // read from shared memory only by first warp
    if (wid == 0){
        // final reduction within first warp
        reduction_warp(val);
    }
}


// integrate over Pi to get P1, *_proc is the MPI proc's coordinate in each direction
__global__ void kernel_p1(double *Pi, double *P1, int iflavor,  int const gsw[4], unsigned VOLUME, int const T_proc, int const LX_proc, int LY_proc, int LZ_proc){
    const int Lmax = get_global_Lmax();
    const int local_dim[4] = {T, LX, LY, LZ};
    const int global_dim[4] = {T_global, LX_global, LY_global, LZ_global};
    const int proc_global[4] = {T_proc, LX_proc, LY_proc, LZ_proc};
    
    // P1[rho][sigma][nu][z[rho]] += Pi[z][sigma][nu]
    // i.e. P1[rho][sigma][nu][i] += sum over all z with z[rho]=i of Pi[z][sigma][nu]
    for (int rho=blockIdx.x; rho<4; rho+=gridDim.x)
    for (int sigma=blockIdx.y; sigma<4; sigma+=gridDim.y)
    for (int nu=blockIdx.z; nu<4; nu+=gridDim.z)
    for (int zr=threadIdx.x; zr<local_dim[rho]; zr+=blockDim.x) {
        double sum = 0.0;
        // the index of the three non-rho directions
        int dir[3];
        int cnt = 0;
        for (int d=0; d<4; d++){
            if (d!=rho){
                dir[cnt] = d;
                cnt++;
            }
        }
        // construct global z[4], rho direction fixed
        int z[4];
        z[rho]=(zr + proc_global[rho] * local_dim[rho]+ global_dim[rho]) % global_dim[rho];


        // loop over the rest VOLUME/local_dim[rho] points in 3D
        for (int iz = 0; iz < VOLUME/local_dim[rho]; iz++) {
            // the 3 other z directions
            z[dir[0]] = (iz / (local_dim[dir[1]] * local_dim[dir[2]]) + proc_global[dir[0]] * local_dim[dir[0]] + global_dim[dir[0]]) % global_dim[dir[0]];
            z[dir[1]] = (iz / local_dim[dir[2]] % local_dim[dir[1]] + proc_global[dir[1]] * local_dim[dir[1]] + global_dim[dir[1]]) % global_dim[dir[1]];
            z[dir[2]] = (iz % local_dim[dir[2]] + proc_global[dir[2]] * local_dim[dir[2]] + global_dim[dir[2]]) % global_dim[dir[2]];

            // now machine address z_lex
            const int z_lex = z[0] * LX_global * LY_global * LZ_global
                      + z[1] * LY_global * LZ_global
                      + z[2] * LZ_global
                      + z[3];

            // accumulate sum
            sum += Pi[z_lex*16 + sigma*4 + nu];
        }

        // write to P1
        int const z_w = (z[rho] - gsw[rho] + global_dim[rho]) % global_dim[rho];
        P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + z_w] = sum;
    }
}


/* pi[volume][4][4][4], P23[n_y][kernel_n][kernel_n_geom][4][4][4] */
__global__ void kernel_p23(double *pi, double *P23, int n_y, int kernel_n, int kernel_n_geom, const int gsw[4], const int *gycoords, const double xunit[2],
/* QED_kernel_temps kqed_t,  */unsigned VOLUME){
    const int n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
    //set_zero(P23, n_p23);
    

    for ( int yi = blockIdx.x; yi < n_y; yi+=gridDim.x ){
        // For P2: y = (gsy - gsw)
        // For P3: y' = (gsw - gsy)
        // We define y = (gsy - gsw) and use -y as input for P3.
        int const * gsy = &gycoords[4*yi];
        int const y[4] = {
            ( gsy[0] - gsw[0] + T_global ) % T_global,
            ( gsy[1] - gsw[1] + LX_global ) % LX_global,
            ( gsy[2] - gsw[2] + LY_global ) % LY_global,
            ( gsy[3] - gsw[3] + LZ_global ) % LZ_global
        };
        int yv[4];
        site_map_zerohalf ( yv, y );

        double const ym[4] = {
            yv[0] * xunit[0],
            yv[1] * xunit[0],
            yv[2] * xunit[0],
            yv[3] * xunit[0] };

        double const ym_minus[4] = {
            -yv[0] * xunit[0],
            -yv[1] * xunit[0],
            -yv[2] * xunit[0],
            -yv[3] * xunit[0] };

        for (int ix = 0; ix<VOLUME; ix ++){
            /* int const x[4] = {
            ( g_lexic2coords[ix][0] + g_proc_coords[0] * T  - gsw[0] + T_global  ) % T_global,
            ( g_lexic2coords[ix][1] + g_proc_coords[1] * LX - gsw[1] + LX_global ) % LX_global,
            ( g_lexic2coords[ix][2] + g_proc_coords[2] * LY - gsw[2] + LY_global ) % LY_global,
            ( g_lexic2coords[ix][3] + g_proc_coords[3] * LZ - gsw[3] + LZ_global ) % LZ_global }; */
            int const x[4] = {(ix / (LX * LY * LZ) - gsw[0] + T_global) % T_global,
            (ix / (LY * LZ) % LX - gsw[1] + LX_global) % LX_global,
            ((ix / LZ) % LY - gsw[2] + LY_global) % LY_global,
            (ix % LZ - gsw[3] + LZ_global) % LZ_global};

            int xv[4];
            site_map_zerohalf ( xv, x );

            const double pix[16] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
                            pi[ix*16 +4], pi[ix*16 +5], pi[ix*16 +6], pi[ix*16 +7],
                            pi[ix*16 +8], pi[ix*16 +9], pi[ix*16 +10],pi[ix*16 +11],
                            pi[ix*16 +12],pi[ix*16 +13],pi[ix*16 +14],pi[ix*16 +15]};
            double const xm[4] = {
            xv[0] * xunit[0],
            xv[1] * xunit[0],
            xv[2] * xunit[0],
            xv[3] * xunit[0] };

            double const xm_minus[4] = {
            -xv[0] * xunit[0],
            -xv[1] * xunit[0],
            -xv[2] * xunit[0],
            -xv[3] * xunit[0] };

            double const xm_mi_ym[4] = {
                xm[0] - ym[0],
                xm[1] - ym[1],
                xm[2] - ym[2],
                xm[3] - ym[3] };
            double const ym_mi_xm[4] = {
                ym[0] - xm[0],
                ym[1] - xm[1],
                ym[2] - xm[2],
                ym[3] - xm[3] };

        // parallelise over ikernel
        for ( int ikernel = threadIdx.x; ikernel < kernel_n; ikernel+=blockDim.x ){
            /* double kerv1[6][4][4][4] KQED_ALIGN ;
            double kerv2[6][4][4][4] KQED_ALIGN ;
            double kerv3[6][4][4][4] KQED_ALIGN ;
            double kerv4[6][4][4][4] KQED_ALIGN ;
            KQED_LX[ikernel]( xm, ym,             kqed_t, kerv1 );
            KQED_LX[ikernel]( ym, xm,             kqed_t, kerv2 );
            KQED_LX[ikernel]( xm_mi_ym, ym_minus, kqed_t, kerv3 );
            KQED_LX[ikernel]( ym_mi_xm, xm_minus, kqed_t, kerv4 ); */
            /* mock value for compilation test*/
            double kerv1[6][4][4][4]={4};
            double kerv2[6][4][4][4]={3};
            double kerv3[6][4][4][4]={2};
            double kerv4[6][4][4][4]={1};

            /* a different local copy of P2/3 for each kernel */
            double local_p2_0[64]={0};
            double local_p2_1[64]={0};
            double local_p3[64]={0};
            
            /* P2_0 unroll k (too much register pressure)*/
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=0: {0,1}
                local_p2_0[0*16 + 1*4 + nu] += kerv1[0][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[0*16 + 1*4 + nu] += kerv2[0][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[0*16 + 1*4 + nu] += kerv3[0][mu][lambda][nu] * pix[mu*4+lambda];
            }
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=1: {0,2}
                local_p2_0[0*16 + 2*4 + nu] += kerv1[1][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[0*16 + 2*4 + nu] += kerv2[1][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[0*16 + 2*4 + nu] += kerv3[1][mu][lambda][nu] * pix[mu*4+lambda];
            }
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=2: {0,3}
                local_p2_0[0*16 + 3*4 + nu] += kerv1[2][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[0*16 + 3*4 + nu] += kerv2[2][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[0*16 + 3*4 + nu] += kerv3[2][mu][lambda][nu] * pix[mu*4+lambda];
            }
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=3: {1,2}
                local_p2_0[1*16 + 2*4 + nu] += kerv1[3][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[1*16 + 2*4 + nu] += kerv2[3][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[1*16 + 2*4 + nu] += kerv3[3][mu][lambda][nu] * pix[mu*4+lambda];
            }
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=4: {1,3}
                local_p2_0[1*16 + 3*4 + nu] += kerv1[4][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[1*16 + 3*4 + nu] += kerv2[4][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[1*16 + 3*4 + nu] += kerv3[4][mu][lambda][nu] * pix[mu*4+lambda];
            }
            for (int mu=0; mu<4; mu++)
            for (int nu=0; nu<4; nu++)
            for (int lambda=0; lambda<4; lambda++){
                // k=5: {2,3}
                local_p2_0[2*16 + 3*4 + nu] += kerv1[5][mu][nu][lambda] * pix[mu*4 +lambda];
                local_p2_1[2*16 + 3*4 + nu] += kerv2[5][nu][mu][lambda] * pix[mu*4+lambda];
                local_p3[2*16 + 3*4 + nu] += kerv3[5][mu][lambda][nu] * pix[mu*4+lambda];
            }

            /* accumulate to global P2/3[yi] */
            double *P23_y = P23 + yi*kernel_n*kernel_n_geom*4*4*4 + ikernel*kernel_n_geom*4*4*4;
            for (int rho=0; rho<4; rho++)
            for (int sigma=0; sigma<4; sigma++)
            for (int nu=0; nu<4; nu++){
                P23_y[0*64 + rho*16 + sigma*4 + nu] += local_p2_0[rho * 16 + sigma * 4 + nu];
                P23_y[1*64 + rho*16 + sigma*4 + nu] += local_p2_1[rho * 16 + sigma * 4 + nu];
                P23_y[2*64 + rho*16 + sigma*4 + nu] += local_p3[rho * 16 + sigma * 4 + nu];
                /* atomicAdd(P23_y + 3*64 + rho*16 + sigma*4 + nu, local_p4_0[rho][sigma][nu]);
                P23[yi][ikernel * kernel_n_geom + 0][rho][sigma][nu] += local_p2_0[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 1][rho][sigma][nu] += local_p2_1[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 2][rho][sigma][nu] += local_p3[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 3][rho][sigma][nu] += local_p4_0[rho][sigma][nu]; */
            }
            __syncthreads();
        }
    }
  }
}

__host__ void compute_2p2_gpu(double * fwd_y, double * P1, double * P23, int iflavor, unsigned VOLUME){
    /* set up problem on gpu */
    double* pi_d, * fwd_y_d, * P1_d, * P23_d;
    cudaMalloc((void **)&pi_d, sizeof(double) * 4 * 4 * 4 * VOLUME);
    cudaMemcpy(fwd_y_d, fwd_y, sizeof(double)* 2 * 12 * 24 * VOLUME, cudaMemcpyHostToDevice);
    /* cudaMalloc((void **)&P1_d, sizeof(double) * 4 * 4 * 4 * T_global);
    cudaMalloc((void **)&P23_d, sizeof(double) * 4 * 4 * 4 * VOLUME); */

    dim3 gridDim(128);
    dim3 blockDim(32);

    /* compute Pi[x][mu][nu] */
    kernel_pi<<<gridDim, blockDim>>>(fwd_y_d, pi_d, iflavor, VOLUME);
    cudaFree(fwd_y_d);

    /* integrate Pi to get P1 */
    const int gsw[4] = {1,2,1,1};
    //kernel_p1<<<gridDim, blockDim>>>(pi_d, P1_d, iflavor, gsw, VOLUME, g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

    const int y[4] = {1, 1, 1, 1};
    int y_coord[80];
    for (int i=0; i<20; i++) {
        y_coord[i*4 + 0] = i * y[0];
        y_coord[i*4 + 1] = i * y[1];
        y_coord[i*4 + 2] = i * y[2];
        y_coord[i*4 + 3] = i * y[3];
    }
    /* Integrate with qed kernels to get P2 and P3 */
    double const unit[2] = {1.2, 1.3};
    //kernel_p23<<<gridDim, blockDim>>>(pi_d, P23_d, 20, kernel_n, kernel_n_geom, gsw, y_coord, unit, VOLUME);

    cudaFree(pi_d);
}

// compare pi computed on cpu and gpu 
// write pi into file named pi_cuda.dat
__host__ void record_pi_cuda(double *fwd_y, int VOLUME, int iflavor) {
    double *pi_gpu = (double *)malloc(4 * 4 * VOLUME * sizeof(double));

    /* set up problem on gpu */
    double* pi_d, * fwd_y_d;
    cudaMalloc((void **)&pi_d, sizeof(double) * 4 * 4 * VOLUME);
    cudaMalloc((void **)&fwd_y_d, sizeof(double)* 2 * 12 * 24 * VOLUME);
    cudaMemcpy(fwd_y_d, fwd_y, sizeof(double)* 2 * 12 * 24 * VOLUME, cudaMemcpyHostToDevice);

    dim3 gridDim(128);
    dim3 blockDim(64);

    // call gpu code
    kernel_pi<<<gridDim, blockDim>>>(fwd_y_d, pi_d, iflavor, VOLUME);
    cudaMemcpy(pi_gpu, pi_d, 4 * 4 * VOLUME * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(fwd_y_d);
    cudaFree(pi_d);

    // write to file
    /* FILE *file;
    for (int i=0; i< 16 * VOLUME; i++) {
        file = fopen("pi_cuda.dat", "a");
        fprintf(file, "%.10e\n", pi_gpu[i]);
        fclose(file);
    } */
    free(pi_gpu);
    return;
}

// write p1 into file named p1_cuda.dat
__host__ void record_p1_cuda(double *Pi, int iflavor, int const * gsw, int VOLUME) {
    // set up on gpu
    double *Pi_d, *P1_d;
    cudaMalloc((void **)&Pi_d, sizeof(double) * 16 * VOLUME);
    cudaMemcpy(Pi_d, Pi, sizeof(double) * 16 * VOLUME, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&P1_d, sizeof(double) * 64 * T_global);

    dim3 gridDim(4, 4, 4);
    dim3 blockDim(64);

    // compute p1 on gpu
    kernel_p1<<<gridDim, blockDim>>>(Pi_d, P1_d, iflavor, gsw, VOLUME, g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);

    // copy back to host
    double *P1 = (double *)malloc(sizeof(double) * 64 * T_global);
    cudaMemcpy(P1, P1_d, sizeof(double) * 64 * T_global, cudaMemcpyDeviceToHost);
    cudaFree(Pi_d);
    cudaFree(P1_d);

    // write to file
    /* FILE *file;
    for (int i=0; i< 64 * T_global; i++) {
        file = fopen("p1_cuda.dat", "a");
        fprintf(file, "%.10e\n", P1[i]);
        fclose(file);
    } */
    free(P1);
    return;
}

__host__ void record_p23_cuda(double *Pi, int n_y, int kernel_n, int kernel_n_geom, const int*gsw, const int *gycoords, const double xunit[2], unsigned VOLUME) {
    // set up on gpu
    double *Pi_d, *P23_d;
    cudaMalloc((void **)&Pi_d, sizeof(double) * 16 * VOLUME);
    cudaMemcpy(Pi_d, Pi, sizeof(double) * 16 * VOLUME, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&P23_d, sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    cudaMemset(P23_d, 0, n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    int *gycoords_d, *gsw_d;
    cudaMalloc((void **)&gycoords_d, sizeof(int) * 4 * n_y);
    cudaMemcpy(gycoords_d, gycoords, sizeof(int) * 4 * n_y, cudaMemcpyHostToDevice);
    //cudaMalloc((void **)&gsw_d, sizeof(int) * 4);
    //cudaMemcpy(gsw_d, gsw, sizeof(int) * 4, cudaMemcpyHostToDevice);

    dim3 gridDim(128);
    dim3 blockDim(kernel_n);

    // compute p23 on gpu
    kernel_p23<<<gridDim, blockDim>>>(Pi_d, P23_d, n_y, kernel_n, kernel_n_geom, gsw, gycoords_d, xunit, VOLUME);

    // copy back to host
    double *P23 = (double *)malloc(sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4);
    cudaMemcpy(P23, P23_d, sizeof(double) * n_y * kernel_n * kernel_n_geom * 4 * 4 *4, cudaMemcpyDeviceToHost);
    cudaFree(Pi_d);
    cudaFree(P23_d);
    cudaFree(gycoords_d);

    // write to file
    /* FILE *file;
    for (int i=0; i< n_y * kernel_n * kernel_n_geom * 4 * 4 *4; i++) {
        file = fopen("p23_cuda.dat", "a");
        fprintf(file, "%.10e\n", P23[i]);
        fclose(file);
    }
    free(P23); */
    return;
}

int main() {
    // allocate fwd_y on the host
    int const VOL = T * LX * LY * LZ;

    //allocate P1, P23 
    //double *P1 = (double *)malloc(sizeof(double) * 4 * 4 * 4 * T_global);
    //double *P23 = (double *)malloc(4*4*4*T_global);

   
    //check Pi correctness
    /* double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) fwd_y[i] = rand()*2./RAND_MAX - 1.;
    record_pi_cuda(fwd_y, VOL, 0);
    free(fwd_y);
 */
    //check P1 correctness
    /* double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    const int gsw[4] = {1, 1, 1, 1};
    record_p1_cuda(Pi, 0, gsw, VOL);
    free(Pi); */   

    //check P23 correctness
    const int n_y = 120;
    const int gsw[4] = {1,1,1,1};
    int *gycoords = (int *)malloc(sizeof(int) * 4 * n_y);
    for (int i=0; i<n_y; i++){
        gycoords[4*i +0] = (i+2)%T_global;
        gycoords[4*i +1] = (i+3)%LX_global;
        gycoords[4*i +2] = (i+4)%LY_global;
        gycoords[4*i +3] = (i+5)%LZ_global;
    }
    double xunit[2] = {0.1,0.2};
    double *Pi = (double *) malloc(sizeof(double) * 16 * VOL);
    srand(1234);
    for (int i=0; i<16 * VOL; i++) Pi[i] = rand()*2./RAND_MAX - 1.;
    record_p23_cuda(Pi, n_y, kernel_n, kernel_n_geom, gsw, gycoords, xunit, VOL);
    free(gycoords);
    free(Pi);

    // probe
    /* const int n_y=20;
    double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    double *P1 = (double *)malloc(64  * T_global * sizeof(double));
    double *P23 = (double *)malloc(n_y * kernel_n * kernel_n_geom * 4 * 4 * 4 *sizeof(double));
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) fwd_y[i] = rand()*2./RAND_MAX - 1.;
    compute_2p2_gpu(fwd_y, P1, P23, 0, VOL);
    free(fwd_y);
    free(P1);
    free(P23); */

    return 0;
}