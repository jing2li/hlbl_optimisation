#ifndef _KERNELS_H
#define _KERNELS_H


#include "kernels.cpp"
//#include "kernels.cu"

/* CPU 2P2 BREAKDOWN, the ones noted with _0 are the original functions */
inline void compute_pi_0(double *** fwd_y, double * Pi, int iflavor, double ** spinor_work, unsigned VOLUME);
inline void compute_pi(double *** fwd_y, double * Pi, int iflavor, unsigned VOLUME);
inline void integrate_p1_0(double * pimn, double *P1, int iflavor, int const * gsw, unsigned VOLUME);
inline void integrate_p1(double *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME);
inline void compute_p23_0(double *pimn, double (*P23)[kernel_n*kernel_n_geom][4][4][4], const int*gsw, int n_y, const int *gycoords, const double xunit[2],
/* QED_kernel_temps kqed_t, */ unsigned VOLUME);
inline void compute_p23(double *pi, double (*P23)[kernel_n*kernel_n_geom][4][4][4], const int*gsw, int n_y, const int *gycoords, const double xunit[2],
/* QED_kernel_temps kqed_t,  */unsigned VOLUME);

/* CPU CHECKS */
void check_Pi(size_t vol);
void check_integral(size_t vol, int w0, int w1, int w2, int w3); // w[4] is gsw
void check_p23(unsigned vol, const int* gsw, int n_y, const int *gycoords, const double xunit[2]);



/* CUDA 2P2 BREAKDOWN */
//void compute_2p2_gpu(double * fwd_y, double * P1, double * P23, int iflavor, unsigned VOLUME);
//void kernel_pi(double* fwd_y, double * Pi, int iflavor, unsigned VOLUME);
//void kernel_p1(double *Pi, double *P1, int iflavor,  int const * gsw, unsigned VOLUME);
//void kernel_p23(double *pi, double *P23, int n_y, int kernel_n, int kernel_n_geom, const int*gsw, const int *gycoords, const double xunit[2],
/* QED_kernel_temps kqed_t,  */ //unsigned VOLUME);

/* CUDA CHECKS */
void check_Pi_cuda();
void check_P1_cuda();
void check_P23_cuda();

#endif