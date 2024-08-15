#ifndef _GPU_INT_ADD_CU_
#define _GPU_INT_ADD_CU_

#include "gpuintadd.cuh"

__global__ void gpuintaddkernel( int a, int b, int* c ) {
  *c = a + b;
}

void gpuintadd( int a, int b, int* c ) {
  int *dev_c;

  cudaMalloc( (void**)&dev_c, sizeof(int) ); // now a pointer to device memory

  gpuintaddkernel<<<1,1>>>(a, b, dev_c);
  
  cudaDeviceSynchronize();

  cudaMemcpy( c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ); // cp memory from device to host

  cudaFree(dev_c);
}

__global__ void kernel() {

}

#endif
