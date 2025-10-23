#include <iostream>
#include <complex>
#include <cmath>
#include <stdlib.h> 
#include "cvc_linalg.h"
//#include <mpi.h>
#include <omp.h>


#define _GSI(x) 24*x
#define T_global 12
#define LX_global 12
#define LY_global 12
#define LZ_global 12

const int T = 12;
const int LX = 12;
const int LY = 12;
const int LZ = 12;

inline int get_Lmax()
{
  int Lmax = 0;
  if ( T_global >= Lmax ) Lmax = T_global;
  if ( LX_global >= Lmax ) Lmax = LX_global;
  if ( LY_global >= Lmax ) Lmax = LY_global;
  if ( LZ_global >= Lmax ) Lmax = LZ_global;
  return Lmax;
}

using namespace cvc;
/* computation of Pi[mu][nu] */
inline void compute_p1_0(double *** fwd_y, double * Pi, int iflavor, double ** spinor_work, unsigned VOLUME) 
{
  double *** pimn = (double ***)malloc(sizeof(double **) *4);
  for (int i=0; i<4; i++){
    pimn[i] = (double **)malloc(sizeof(double *) *4);
    for (int j=0; j<4; j++){
      pimn[i][j] = (double *)calloc(VOLUME, sizeof(double));
    }
  }
  /* memset((void*)pimn[0][0], 0, sizeof(double)*4*4*VOLUME); */
  for (int i=0; i<4; i++)
  for (int j=0; j<4; j++)
  for (size_t k=0; k<VOLUME; k++){
    pimn[i][j][k] = 0.0;
  }
  
  for ( int nu = 0; nu < 4; nu++ )
  {
    for ( int mu = 0; mu < 4; mu++ )
    {
      for ( int ia = 0; ia < 12; ia++ )
      {
#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++ )
        {
          const double * _u = fwd_y[iflavor][ia] + _GSI(ix);
          double * _t = spinor_work[0] + _GSI(ix);
          _fv_eq_gamma_ti_fv ( _t, mu, _u );
          _fv_ti_eq_g5 ( _t );
          double * _s = spinor_work[1] + _GSI(ix);
          for ( int ib = 0; ib < 12; ib++ )
          {
            const double * _d = fwd_y[1-iflavor][ib] + _GSI(ix);
            complex w;
            _co_eq_fv_dag_ti_fv ( &w, _d, _t );
            _s[2*ib]   = w.re;
            _s[2*ib+1] = w.im;
          }
          _fv_ti_eq_g5 ( _s );
          _fv_eq_gamma_ti_fv ( _t, nu, _s );
          // real part
          pimn[mu][nu][ix] += _t[2*ia];
        }
      }
    }
  }

  /* copy to p1 */
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int ix=0; ix<VOLUME; ix++) {
    Pi[mu*4*VOLUME + nu*VOLUME + ix] = pimn[mu][nu][ix];
  }

}

/* performance improvement version 1: rearrange data structure of pi */
inline void compute_p1_1(double *** fwd_y, double * Pi, int iflavor, unsigned VOLUME) 
{
  double * pi = (double *)calloc((size_t)VOLUME *  4 * 4, sizeof(double)); /* pi[x][mu][nu] */

  /* loop over position volume */
  #pragma omp parallel for
  for (int ix = 0; ix < VOLUME; ix++) {
    /* load 12 x 12 (x2 complex) d and u from fwd_y*/
    double u[12][24];
    double d[12][24];
    for (int ia = 0; ia < 12; ia++)
    for (int ib = 0; ib < 24; ib++) {
      u[ia][ib] = fwd_y[iflavor][ia][_GSI(ix) + ib];
    }
    for (int ia = 0; ia < 12; ia++)
    for (int ib = 0; ib < 24; ib++){
      d[ia][ib] = fwd_y[1-iflavor][ia][_GSI(ix) + ib];
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
      pi[ix*16+mu*4+nu] = trace;
    }
  }

  /* copy to p1 */
  for (int ix=0; ix<VOLUME; ix++)
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++) {
    Pi[mu*4*VOLUME + nu*VOLUME + ix] = pi[ix*16 + mu*4 + nu];
  }

}


/* Integration of Pi[mu][nu] over z */
void integrate_p1_0(double * pimn, double *P1, int iflavor, int const * gsw, unsigned VOLUME) 
{
  const int Lmax = get_Lmax();
  const int n_P1 = 4 * 4 * 4 * Lmax;
  //double **** local_P1 = init_4level_dtable ( 4, 4, 4, Lmax );
  double **** local_P1 = (double ****)malloc(sizeof(double ***) *4);
  for (int i=0; i<4; i++){
    local_P1[i] = (double ***)malloc(sizeof(double **) *4);
    for (int j=0; j<4; j++){
      local_P1[i][j] = (double **)malloc(sizeof(double *) *4);
      for (int k=0; k<4; k++){
        local_P1[i][j][k] = (double *)calloc(Lmax, sizeof(double));
      }
    }
  }
  if ( local_P1 == NULL )
  {
    fprintf ( stderr, "Error alloc local_P1\n" );
    exit ( 57 );
  }
  //memset((void*)local_P1[0][0][0], 0, sizeof(double)*n_P1);

  for ( int sigma = 0; sigma < 4; sigma++ )
  {
    for ( int nu = 0; nu < 4; nu++ )
    {
      // TODO: Parallelize over non-summed coordinate?
      for ( unsigned int iz = 0; iz < VOLUME; iz++ )
      {
        /* int const z[4] = {
          ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - gsw[0] + T_global  ) % T_global,
          ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - gsw[1] + LX_global ) % LX_global,
          ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - gsw[2] + LY_global ) % LY_global,
          ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - gsw[3] + LZ_global ) % LZ_global }; 
        */
        
          /* find global z[4] */
        const int z[4] = {(iz / (LX * LY * LZ) - gsw[0] + T_global) % T_global,
          (iz / (LY * LZ) % LX - gsw[1] + LX_global) % LX_global,
          ((iz / LZ) % LY - gsw[2] + LY_global) % LY_global,
          (iz % LZ - gsw[3] + LZ_global) % LZ_global};

        for ( int rho = 0; rho < 4; rho++ )
        {
          local_P1[rho][sigma][nu][z[rho]] += pimn[sigma*4*VOLUME + nu*VOLUME + iz];
        }
      }
    }
  }

  for (int rho=0; rho<4; rho++)
  for (int sigma=0; sigma<4; sigma++)
  for (int nu=0; nu<4; nu++)
  for (int i=0; i<Lmax; i++) {
    P1[rho*16*Lmax + sigma*4*Lmax + nu*Lmax + i] = local_P1[rho][sigma][nu][i];
  }

  /* if ( MPI_Allreduce(local_P1[0][0][0], P1, n_P1, MPI_DOUBLE, MPI_SUM, g_cart_grid)
      != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
  } */

}

/* rerarrange the summation order to z, rho, sigma, nu
   note that input pi[x][mu][nu] is different P1[rho][sigma][nu][z] is unchanged */
void integrate_p1_1(double *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME) 
{
  const int Lmax = get_Lmax();
  const int n_P1 = 4 * 4 * 4 * Lmax;
  //double **** local_P1 = init_4level_dtable ( 4, 4, 4, Lmax );
  double * local_P1 = (double *)calloc(n_P1, sizeof(double));

  if ( local_P1 == NULL )
  {
    fprintf ( stderr, "Error alloc local_P1\n" );
    exit ( 57 );
  }

  for (int iz = 0; iz < VOLUME; iz++ ) {
    /* find global z[4] */
    const int z[4] = {(iz / (LX * LY * LZ) - gsw[0] + T_global) % T_global,
      (iz / (LY * LZ) % LX - gsw[1] + LX_global) % LX_global,
      ((iz / LZ) % LY - gsw[2] + LY_global) % LY_global,
      (iz % LZ - gsw[3] + LZ_global) % LZ_global};

    for (int rho=0; rho<4; rho++)
    for (int sigma=0; sigma<4; sigma++)
    for (int nu=0; nu<4; nu++) {
        local_P1[rho*16*Lmax + sigma*Lmax*4 + nu*Lmax + z[rho]] += Pi[iz*16 + sigma*4 + nu];
    }
    
  }

  /* copy to P1 */
  for (int i=0; i<n_P1; i++) {
    P1[i] = local_P1[i];
  }
  /* if ( MPI_Allreduce(local_P1[0][0][0], P1, n_P1, MPI_DOUBLE, MPI_SUM, g_cart_grid)
      != MPI_SUCCESS ) {
    if ( g_cart_id == 0 ) fprintf ( stderr, "[] Error from MPI_Allreduce %s %d\n", __FILE__, __LINE__ );
  } */

}

/* check correctness of P1 */
void check_Pi(size_t vol) {
  double *p1_0 = (double *)calloc(16 * vol, sizeof(double));
  double *p1_1 = (double *)calloc(16 * vol, sizeof(double));
  //double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  double **spinor_work = (double **) malloc(sizeof(double *) * 2);
  for (int i=0; i<2; i++) spinor_work[i] = (double *)malloc(sizeof(double)*_GSI(vol));
  //double *** fwd_y = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  double *** fwd_y = (double ***) malloc(sizeof(double **)*2);
  for (int i=0; i<2; i++) {
    fwd_y[i] = (double **) malloc(sizeof(double *) * 12);
    for (int j=0; j<12; j++)
      fwd_y[i][j] = (double *) malloc(sizeof(double) * _GSI(vol));
  }

  /* fill fwd_y with test data ... */
  for (int iflavor = 0; iflavor < 2; iflavor++) {
    srand(iflavor + 1234);
    #pragma omp parallel for collapse(3)
    for (int ia = 0; ia < 12; ia++) {
      for (unsigned int ix = 0; ix < vol; ix++) {
        for (int comp = 0; comp < 24; comp++) {
          fwd_y[iflavor][ia][_GSI(ix) + comp] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
        }
      }
    }
  } 

  compute_p1_0(fwd_y, p1_0, 0, spinor_work, vol);

  compute_p1_1(fwd_y, p1_1, 0, vol);

  int flag = 0;
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int x=0; x<vol; x++){
    const double p0 = p1_0[mu*4*vol+nu*vol+x];
    const double p1 = p1_1[mu*4*vol+nu*vol+x];
    const double diff = p0 - p1;
    if (diff * diff > 1e-20) {
      flag=1;
      printf("P1 difference at [%d][%d][%d], %f VS %f\n.", mu, nu, x, p0, p1);
    }
  }
  if (flag) printf("P1 correctnenss FAILED.\n");
  else printf("P1 correctness PASSED.\n");

  free(p1_0);
  free(p1_1);
  for (int i=0; i<2; i++) free(spinor_work[i]);
  free(spinor_work);
  for (int i=0; i<2; i++){
    for (int j=0; j<12; j++){
      free(fwd_y[i][j]);
    }
    free(fwd_y[i]);
  }
  free(fwd_y);
}
void check_integral(size_t vol, int w0, int w1, int w2, int w3) {
  double *pi = (double *)calloc(16 * vol, sizeof(double));
  double *pi_rearrange = (double *)calloc(16 * vol, sizeof(double));
  double *P1 = (double *)calloc(4 * 4 * 4 * get_Lmax(), sizeof(double));
  for (int i=0; i<16*vol; i++){
    pi[i] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
  }
  const int gsw[4] = {w0,w1,w2,w3};
  integrate_p1_0(pi, P1, 0, gsw, vol);
  double *P1_check = (double *)calloc(4 * 4 * 4 * get_Lmax(), sizeof(double));
  /* Rearrange pi for new format i.e. pi[mu][nu][x] to pi[x][mu][nu] */
  for (int mu=0; mu<4; mu++)
  for (int nu=0; nu<4; nu++)
  for (int x=0; x<vol; x++){
    pi_rearrange[x*16 + mu*4 + nu] = pi[mu*4*vol + nu*vol + x];
  }

  integrate_p1_1(pi_rearrange, P1_check, 0, gsw, vol);
  int flag = 0;
  const int n_P1 = 4 * 4 * 4 * get_Lmax();
  for (int i=0; i<n_P1; i++){
    const double diff = P1[i] - P1_check[i];
    if (diff * diff > 1e-20) {
      flag=1;
      printf("Integral difference at %d: %f VS %f\n.", i, P1[i], P1_check[i]);
    }
    //printf("Integral difference at %d: %f VS %f\n.", i, P1[i], P1_check[i]);
  }
  if (flag) printf("Integral correctness FAILED.\n");
  else printf("Integral correctness PASSED.\n");
  free(pi);
  free(pi_rearrange);
  free(P1);
  free(P1_check);
}

int main ( int argc, char **argv )
{
  size_t VOLUME = LX_global * LY_global * LZ_global * T_global;
  size_t RAND = 0;
  const int Lmax = get_Lmax();
  init_gamma();
  double *p1_0 = (double *)calloc(16 * VOLUME, sizeof(double));
  double *p1_1 = (double *)calloc(16 * VOLUME, sizeof(double));
  //double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  double **spinor_work = (double **) malloc(sizeof(double *) * 2);
  for (int i=0; i<2; i++) spinor_work[i] = (double *)malloc(sizeof(double)*_GSI(VOLUME));
  //double *** fwd_y = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  double *** fwd_y = (double ***) malloc(sizeof(double **)*2);
  for (int i=0; i<2; i++) {
    fwd_y[i] = (double **) malloc(sizeof(double *) * 12);
    for (int j=0; j<12; j++)
      fwd_y[i][j] = (double *) malloc(sizeof(double) * _GSI(VOLUME));
  }

  //double ***** P1 = init_5level_dtable ( 2, 4, 4, 4, Lmax );
  double * P1 = (double *) malloc (sizeof(double) * 4 * 4 * 4 * Lmax);

  /* fill fwd_y with test data ... */
  for (int iflavor = 0; iflavor < 2; iflavor++) {
    srand(iflavor + 1234);
    #pragma omp parallel for collapse(3)
    for (int ia = 0; ia < 12; ia++) {
      for (unsigned int ix = 0; ix < VOLUME; ix++) {
        for (int comp = 0; comp < 24; comp++) {
          fwd_y[iflavor][ia][_GSI(ix) + comp] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
        }
      }
    }
  } 

  /* compute_p1_0(fwd_y, p1_0, 0, spinor_work, VOLUME); */

  /* compute_p1_1(fwd_y, p1_1, 0, VOLUME); */

  const int src[4] = {0, 0, 0, 0};
  /* integrate_p1_0(p1_0, P1, 0, src, VOLUME); */
  
  /* check_Pi(VOLUME); */
  check_integral(VOLUME, src[0], src[1], src[2], src[3]);

  free(p1_0);
  free(p1_1);
  for (int i=0; i<2; i++) free(spinor_work[i]);
  free(spinor_work);
  for (int i=0; i<2; i++){
    for (int j=0; j<12; j++){
      free(fwd_y[i][j]);
    }
    free(fwd_y[i]);
  }
  free(fwd_y);
  free(P1);

  return 0;
}

