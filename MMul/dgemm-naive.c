/*
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

const char* dgemm_desc = "Naive, three-loop dgemm.";
#include "immintrin.h"
#include<stdio.h>
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm (int n, double* A, double* B, double*__restrict__ C)
{
  register int i = 0; register int k = 0; register int j = 0; int c = 0;
      for(i = 0; i < n; i+=16)
       {
        c = i;
        if((i+15) >= n)
        {
         for(i = c; i < n; i++)
         {
          for(j = 0; j< n; j++)
          {
           for(k = 0; k < n; k++)
            C[i+j*n]+= A[i+k*n]*B[k+j*n];
          }
         }
         break;
        }
         for(j = 0; j < n; j++)
         {
          __m256d cij = _mm256_load_pd(C+i+j*n);
          __m256d cji = _mm256_load_pd(C+(i+4)+j*n);
          __m256d coo = _mm256_load_pd(C+(i+8)+j*n);
          __m256d wji = _mm256_load_pd(C+(i+12)+j*n);
          for(k = 0; k < n; k++)
          {
            cij =_mm256_add_pd(cij, _mm256_mul_pd(_mm256_load_pd(A+i+k*n),_mm256_broadcast_sd(B+k+j*n)));
            cji =_mm256_add_pd(cji, _mm256_mul_pd(_mm256_load_pd(A+(i+4)+k*n),_mm256_broadcast_sd(B+k+j*n)));
            coo =_mm256_add_pd(coo, _mm256_mul_pd(_mm256_load_pd(A+(i+8)+k*n),_mm256_broadcast_sd(B+k+j*n)));
            wji =_mm256_add_pd(wji, _mm256_mul_pd(_mm256_load_pd(A+(i+12)+k*n),_mm256_broadcast_sd(B+k+j*n)));
          }
          _mm256_store_pd(C+i+j*n, cij);
          _mm256_store_pd(C+(i+4)+j*n, cji);
          _mm256_store_pd(C+(i+8)+j*n, coo);
          _mm256_store_pd(C+(i+12)+j*n, wji);
        }
       }


}
