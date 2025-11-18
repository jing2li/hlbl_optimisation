#include "kernels.h"

int main ( int argc, char **argv )
{
  // set up MPI cartesian
  /* int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  printf("MPI thread support level: %d\n", provided);
  if (provided < MPI_THREAD_FUNNELED){
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1; // Usually not reached
  }
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int const proc_dim[4] = {NPROCT, NPROCX, NPROCY, NPROCZ};
  const int period[4] = {0,0,0,0};
  MPI_Cart_create(MPI_COMM_WORLD, 4, proc_dim, period, true, &g_cart_grid);
  MPI_Comm_rank(g_cart_grid, &g_cart_id);
  MPI_Cart_coords(g_cart_grid, g_cart_id, 4, g_proc_coords); */


  init_gamma();
  int const VOLUME = LX * LY * LZ * T; // local volume
  int const RAND = 0;
  const int Lmax = get_Lmax();
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
    for (int ia = 0; ia < 12; ia++) {
      for (unsigned int ix = 0; ix < VOLUME; ix++) {
        for (int comp = 0; comp < 24; comp++) {
          fwd_y[iflavor][ia][_GSI(ix) + comp] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
        }
      }
    }
  } 

  /* fill p1_1 with test data  */
  /* for (int i=0; i<16*VOLUME; i++){
    p1_1[i] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
  } */

  /* compute_pi_0(fwd_y, p1_0, 0, spinor_work, VOLUME); */

  //compute_pi(fwd_y, p1_1, 0, VOLUME);

  const int src[4] = {0, 0, 0, 0};
  /* integrate_p1_0(p1_0, P1, 0, src, VOLUME); */
  /* integrate_p1_1(p1_1, P1, 0, src, VOLUME); */

  const int y[4] = {1, 1, 1, 1};
  int y_coord[80];
  for (int i=0; i<20; i++) {
    y_coord[i*4 + 0] = i * y[0];
    y_coord[i*4 + 1] = i * y[1];
    y_coord[i*4 + 2] = i * y[2];
    y_coord[i*4 + 3] = i * y[3];
  }
  const double xunit[2] = {1.0, 2.1};
  double (*P23)[kernel_n*kernel_n_geom+64][4][4][4] = (double (*)[kernel_n*kernel_n_geom+64][4][4][4]) malloc(sizeof(*P23) * 20);
  /* fill p1_1 with test data */
  for (int i=0; i<16*VOLUME; i++){
    p1_1[i] = rand() * 2. / RAND_MAX - 1; // a random number between -1 and 1
  }
  //compute_p23_0(p1_1, P23, src, 20, (const int *)y_coord, xunit, VOLUME);
  //compute_p23(p1_1, P23, src, 20, (const int *)y_coord, xunit, VOLUME);
  
  //check_Pi(VOLUME);
  //check_integral(VOLUME, src[0], src[1], src[2], src[3]);
  //check_p23(VOLUME, src, 10, (const int *)y_coord, xunit);

  //check_Pi_cuda();
  //check_P1_cuda();
  check_P23_cuda();

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

  //MPI_Finalize();

  return 0;
}

