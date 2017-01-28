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
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
#define UNROLL (4)
#define BLOCKS 32
void do_block(int n, int si, int sj, int sk, double *A, double *B, double *C)
{
  for(int i = si; i < (si+BLOCKS); i+=(UNROLL*4))
   for(int j = sj; j < (sj+BLOCKS); j++)
   {
     __m256d c[4];
     for(int x = 0; x < UNROLL; x++)
      c[x] = _mm256_load_pd(C+(i+(x*4))+j*n);
     for(int k = sk; k < sk+BLOCKS; k++)
     {
        __m256d b = _mm256_broadcast_sd(B+k+j*n);
        for(int x = 0; x < UNROLL; x++)
         c[x] = _mm256_add_pd(c[x], _mm256_mul_pd(_mm256_load_pd(A+n*k+((x*4)+i)), b));
     }

     for(int x = 0; x < UNROLL; x++)
      _mm256_store_pd(C+(i+(x*4))+j*n, c[x]);
   }
}






void square_dgemm (int n, double* A, double* B, double* C)
{
 if(n % 32 == 0)
 { for(int sj = 0; sj < n; sj += BLOCKS)
    for(int si = 0; si < n; si += BLOCKS)
     for(int sk = 0; sk < n; sk += BLOCKS)
      do_block(n, si, sj, sk, A, B, C);
 }
 else
 {
  for(int i = 0; i < n; i++)
  {
   for(int j = 0; j < n; j++)
   {
    for(int k = 0; k < n; k++)
    {
     C[i+j*n]+= A[i+k*n] * B[k+j*n];
    }
   }
  }
 }
}
