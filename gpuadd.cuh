#ifndef _GPU_ADD_CUH_
#define _GPU_ADD_CUH_

__global__ void gpuadd( double *a, double *b, double *c, int n);
__global__ void gpumatadd( double *a, double *b, double *c, int n, int n1);

#endif
