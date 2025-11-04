# hlbl_optimisation
Some optimised kernels for hadronic light by light calculations in cvc. The CPU version has improved data layout and better OpenMP and MPI parallelisation. Correctness tests i.e. comparisons to the original are included.
GPU version is in working.

2 + 2 disconnected calculations:
1. Computation Pi (contraction)
   rearranged the data layout and loop order to expose the data parallelism in the problem.
2. Copmutation of P1 (integral of volume)
    benefits the same way as pi from the adjusted data layout
3. computation of P2 and P3 (loop over n_y points and integral over volume)
   parallised with omp threads over y since they are independent calculations. Thread local arrays in an attempt to reduce sharing (? need to verify effectiveness)
