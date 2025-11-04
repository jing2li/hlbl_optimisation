/* GPU kernels of 2p2
    JingJing Li, 2025
*/
#include <iostream>
#include "cvc_complex.h"
//#include <cmath>
#include <stdio.h> 
#include <stdlib.h>
//#include "cvc_linalg.h"
#include <vector>
#include "global.h"

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
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


__device__ inline static int get_Lmax()
{
  int Lmax = 0;
  if ( T_global >= Lmax ) Lmax = T_global;
  if ( LX_global >= Lmax ) Lmax = LX_global;
  if ( LY_global >= Lmax ) Lmax = LY_global;
  if ( LZ_global >= Lmax ) Lmax = LZ_global;
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


using namespace cvc;
//using namespace std;

__global__ void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned VOLUME){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int ix = tid; ix < VOLUME; ix += gridDim.x * blockDim.x) {
        /* load 12 x 12 (x2 complex) d and u from fwd_y*/
        double u[12][24];
        double d[12][24];
        for (int ia = 0; ia < 12; ia++)
        for (int ib = 0; ib < 24; ib++) {
            u[ia][ib] = fwd_y[prop_idx(iflavor, ia, ix, ib)];
        }
        for (int ia = 0; ia < 12; ia++)
        for (int ib = 0; ib < 24; ib++){
            d[ia][ib] = fwd_y[prop_idx(1-iflavor, ia, ix, ib)];
        }

        /* loop over mu and nu */
        for (int mu=0; mu<4; mu++)
        for (int nu=0; nu<4; nu++) {
        double gu[12][24];
        double dot_prod[12][24];

        /* apply gammas: gu = g_5 g_mu u */
        for (int ia=0; ia<12; ia++) {
            _fv_eq_gamma_ti_fv(gu[ia], mu, u[ia]);
            _fv_ti_eq_g5(gu[ia]);
        }

        /* compute <d, u>, dot_prod = <d, gmu> */
        for (int ia=0; ia<12; ia++ ) 
        for (int ib=0; ib<12; ib++ ) {
            complex w;
            _co_eq_fv_dag_ti_fv(&w, d[ib], gu[ia]);
            dot_prod[ia][2*ib] = w.re;
            dot_prod[ia][2*ib + 1] = w.im;
        }

        /* apply gammas: gu = g_nu g_5 dot_prod */
        for (int ia=0; ia<12; ia++) {
            _fv_ti_eq_g5(dot_prod[ia]);
            _fv_eq_gamma_ti_fv(gu[ia], nu, dot_prod[ia]);
        }

        /* take trace over ia: pi[x][mu][nu] = Tr[gu] */
        double trace = 0.0;
        for (int ia=0; ia<12; ia++) {trace += gu[ia][2 * ia]; } // only real part
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


__global__ void kernel_p1(double *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int Lmax = T_global; // T will be the largest dimension
    const int n_P1 = 4 * 4 * 4 * Lmax;

    __shared__ double local_P1[n_P1];
    set_zero(local_P1, n_P1);
    __syncthreads(); 
    for (int iz = tid; iz < VOLUME; iz += gridDim.x * blockDim.x) {
    /* find global z[4] */
        const int z[4] = {(iz / (LX * LY * LZ) - gsw[0] + T_global) % T_global,
            (iz / (LY * LZ) % LX - gsw[1] + LX_global) % LX_global,
            ((iz / LZ) % LY - gsw[2] + LY_global) % LY_global,
            (iz % LZ - gsw[3] + LZ_global) % LZ_global};
            
        for (int rho=0; rho<4; rho++)
        for (int sigma=0; sigma<4; sigma++)
        for (int nu=0; nu<4; nu++) {
            local_P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + z[rho]] 
                += Pi[iz*16 + sigma*4 + nu]; 
        }
    }
    __syncthreads(); // make sure local_P1 computation is complete

    /* copy local to global P1 */
    for (int rho=0; rho<4; rho++)
    for (int sigma=0; sigma<4; sigma++)
    for (int nu=0; nu<4; nu++)
    for (int z=0; z<Lmax; z++){
        //P1[rho] += local_P1[rho];
        atomicAdd(P1+rho*16*Lmax + sigma*4*Lmax + nu*Lmax + z, local_P1[rho]);
    }
}


/* pi[volume][4][4][4], P23[n_y][kernel_n][kernel_n_geom][4][4][4] */
__global__ void kernel_p23(double *pi, double *P23, int n_y, int kernel_n, int kernel_n_geom, const int*gsw, const int *gycoords, const double xunit[2],
/* QED_kernel_temps kqed_t,  */unsigned VOLUME){
    const int n_p23 = n_y * kernel_n * kernel_n_geom * 4 * 4 * 4;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    set_zero(P23, n_p23);

    for ( int yi = 0; yi < n_y; yi++ ){
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

        //parallelise over x 
        for (int ix = tid; ix<VOLUME; ix += blockDim.x * gridDim.x){
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

            const double pix[4][4] = {pi[ix*16 +0], pi[ix*16 +1], pi[ix*16 +2], pi[ix*16 +3],
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

        for ( int ikernel = 0; ikernel < kernel_n; ikernel++ ){
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
            double local_p2_0[4][4][4]={0};
            double local_p2_1[4][4][4]={0};
            double local_p3[4][4][4]={0};
            double local_p4_0[4][4][4]={0};
            double local_p4_1[4][4][4]={0};

            /* P2_0 */
            for (int k=0; k<6; k++){
                const int rho = idx_comb_d[k][0];
                const int sigma = idx_comb_d[k][1];
                for (int mu=0; mu<4; mu++)
                for (int nu=0; nu<4; nu++)
                for (int lambda=0; lambda<4; lambda++){
                    local_p2_0[rho][sigma][nu] += kerv1[k][mu][nu][lambda] * pix[mu][lambda];
                }
            }
            /* P2_1 */
            for (int k=0; k<6; k++){
                const int rho = idx_comb_d[k][0];
                const int sigma = idx_comb_d[k][1];
                for (int nu=0; nu<4; nu++)
                for (int mu=0; mu<4; mu++)
                for (int lambda=0; lambda<4; lambda++){
                    local_p2_1[rho][sigma][nu] += kerv2[k][nu][mu][lambda] * pix[mu][lambda];
                }
            }
            
            /* P3 */
            for (int k=0; k<6; k++){
                const int rho = idx_comb_d[k][0];
                const int sigma = idx_comb_d[k][1];
                for (int mu=0; mu<4; mu++)
                for (int lambda=0; lambda<4; lambda++)
                for (int nu=0; nu<4; nu++){
                    local_p3[rho][sigma][nu] += kerv3[k][mu][lambda][nu] * pix[mu][lambda];
                }
            }
            /* P4_0 */
            /* for (int k=0; k<6; k++){
                const int rho = idx_comb_d[k][0];
                const int sigma = idx_comb_d[k][1];
                for (int nu=0; nu<4; nu++)
                for (int lambda=0; lambda<4; lambda++)
                for (int mu=0; mu<4; mu++){
                    local_p4_0[rho][sigma][nu] += kerv4[k][nu][lambda][mu] * pix[mu][lambda];
                }
            } */
            /* P4_1 */
            /* for( int k = 0; k < 6; k++ )
            {
            const int rho   = idx_comb_d[k][0];
            const int sigma = idx_comb_d[k][1];
            for ( int nu = 0; nu < 4; nu++ ){
                local_p4_1[rho][sigma][nu] = (yv[rho]-xv[rho]) * local_p4_0[rho][sigma][nu];
                local_p4_1[sigma][rho][nu] = (yv[sigma]-xv[sigma]) * (-local_p4_0[rho][sigma][nu]);
            }
            } */
            /* accumulate to global P2/3[yi] */
            double *P23_y = P23 + yi*ikernel*kernel_n_geom*4*4*4;
            for (int rho=0; rho<4; rho++)
            for (int sigma=0; sigma<4; sigma++)
            for (int nu=0; nu<4; nu++){
                atomicAdd(P23_y + 0*64 + rho*16 + sigma*4 + nu, local_p2_0[rho][sigma][nu]);
                atomicAdd(P23_y + 1*64 + rho*16 + sigma*4 + nu, local_p2_1[rho][sigma][nu]);
                atomicAdd(P23_y + 2*64 + rho*16 + sigma*4 + nu, local_p3[rho][sigma][nu]);
                /* atomicAdd(P23_y + 3*64 + rho*16 + sigma*4 + nu, local_p4_0[rho][sigma][nu]);
                P23[yi][ikernel * kernel_n_geom + 0][rho][sigma][nu] += local_p2_0[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 1][rho][sigma][nu] += local_p2_1[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 2][rho][sigma][nu] += local_p3[rho][sigma][nu];
                P23[yi][ikernel * kernel_n_geom + 3][rho][sigma][nu] += local_p4_0[rho][sigma][nu]; */
            }
        }
    }
  }
}


__host__ void compute_2p2_gpu(double * fwd_y, double * P1, double * P23, int iflavor, unsigned VOLUME){
    /* set up problem on gpu */
    double* pi_d, * fwd_y_d, * P1_d, * P23_d;
    cudaMalloc((void **)&pi_d, sizeof(double) * 4 * 4 * 4 * VOLUME);
    cudaMemcpy(fwd_y_d, fwd_y, sizeof(double)* 2 * 12 * 24 * VOLUME, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&P1_d, sizeof(double) * 4 * 4 * 4 * T_global);
    cudaMalloc((void **)&P23_d, sizeof(double) * 4 * 4 * 4 * VOLUME);

    dim3 gridDim(16);
    dim3 blockDim(32);

    /* compute Pi[x][mu][nu] */
    kernel_pi<<<gridDim, blockDim>>>(fwd_y_d, pi_d, iflavor, VOLUME);
    cudaFree(fwd_y_d);

    /* integrate Pi to get P1 */
    //kernel_p1<<<gridDim, blockDim>>>();

    /* Integrate with qed kernels to get P2 and P3 */
    //kernel_p23<<<gridDim, blockDim>>>();

    cudaFree(pi_d);
}

int main() {
    // allocate fwd_y on the host
    int const VOL = T * LX * LY * LZ;
    double *fwd_y = (double *)malloc(2 * 12 * _GSI(VOL) * sizeof(double));
    srand(1234);
    for (int i=0; i<24 * _GSI(VOL); i++) fwd_y[i] = rand()*2./RAND_MAX - 1.;

    //allocate P1, P23 
    double *P1 = (double *)malloc(sizeof(double) * 4 * 4 * 4 * T_global);
    double *P23 = (double *)malloc(4*4*4*T_global);

    // call gpu code
    compute_2p2_gpu(fwd_y, P1, P23, 0, VOL);
}