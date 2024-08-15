#ifndef _GPU_INT_ADD_CUH_
#define _GPU_INT_ADD_CUH_

void gpuintadd( int a, int b, int* c);

__global__ void gpuintaddkernel( int a, int b, int* c);

__global__ void kernel();

#endif 
