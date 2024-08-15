#ifndef _GPU_ADD_CU_
#define _GPU_ADD_CU_

#include <cstring>
#include <iostream>
#include <stdio.h>

__global__ void gpuvectoraddkernel( double *d_A, double *d_B, double *d_C, size_t n) {
  //printf("in add kernel\n");
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  if (x < n) d_C[x] = d_A[x] + d_B[x];
}

__global__ void gpuaddkernel( double *d_A, double *d_B, double *d_C, size_t n, size_t n1) {
  //printf("in add kernel\n");
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = x + n1 * y;
  if (x < n1 && y < n) d_C[idx] = d_A[idx] + d_B[idx];
}


void gpuadd( double *a, double *b, double *c, int n) {

  double *d_A = NULL;
  cudaMalloc( (void**) &d_A, n * sizeof(double) );
  cudaMemcpy(d_A, a, n * sizeof(double), cudaMemcpyHostToDevice); 

  double *d_B = NULL;
  cudaMalloc( (void**) &d_B, n * sizeof(double) );
  cudaMemcpy(d_B, b, n * sizeof(double), cudaMemcpyHostToDevice); 

  double *d_C = NULL;
  cudaMalloc( (void**) &d_C, n * sizeof(double) );

  std::cout <<  "...on GPU, still a loop for now" << std::endl;

  dim3 block = dim3(32,8,1); // 32*8*1 = 256 threads
  dim3 grid = dim3( (n + block.x - 1) / block.x, 1, 1 ); // only 1-d, vector addition
  gpuvectoraddkernel<<<block,grid>>>(d_A, d_B, d_C, n);
  //for (int i = 0; i < n; ++i) 
  //  d_C[i] = d_A[i] + d_B[i];

  std::cout << " finished sum loop" << std::endl;

  cudaDeviceSynchronize();

  //double *C_gpu = new double[n];
  cudaMemcpy(c, d_C, n * sizeof(double), cudaMemcpyDeviceToHost);

  //std::memcpy( (void*)c, (void*)C_gpu, n * sizeof(double) );

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  //free(C_gpu);
}

void printgpuinfo() {
  cudaDeviceProp deviceProp;

  int count;
  cudaGetDeviceCount( &count ) ;
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties( &deviceProp, i);
    std::cout << "  --- General Information for device " << i << " ---  \n";
    std::cout << "Name: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    std::cout << "Clock rate: " << deviceProp.clockRate << std::endl;
    std::cout << "Device copy overlap: " << (deviceProp.deviceOverlap ? " Enabled\n" : " Disabled\n");
    std::cout << "Kernel = execution timeout: " << (deviceProp.kernelExecTimeoutEnabled ? " Enabled\n" : " Disabled\n");
    std::cout << "  --- Memory Information for device " << i << " ---  \n";
    std::cout << "Total global mem: " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "Total constant mem: " << deviceProp.totalConstMem << std::endl; 
    std::cout << "Max mem pitch: " << deviceProp.memPitch << std::endl;
    std::cout << "Texture alignment: " << deviceProp.textureAlignment << std::endl;
    std::cout << "  --- MP Information for device " << i << " ---  \n";
    std::cout << "Multiprocessor count " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Shared mem per mp: " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Registers per mp: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Threads in warp: " << deviceProp.warpSize << std::endl;
    std::cout << "Max threads per block " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads dimensions: (" << deviceProp.maxThreadsDim[0] << "," << deviceProp.maxThreadsDim[1]
        << "," << deviceProp.maxThreadsDim[2] << ")\n";
    std::cout << std::endl;

  }
}

void gpumatadd( double *a, double *b, double *c, int n, int n1) {

  printf("in gpumat add function");
  printgpuinfo();

  cudaDeviceProp deviceProp;
  memset( &deviceProp, 0, sizeof(deviceProp) );
  if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0) ) {
    printf( "\n%s", cudaGetErrorString( cudaGetLastError() ) );
    return;
  }

  

  dim3 block = dim3(32,8,1); // 32*8*1 = 256 threads
  dim3 grid = dim3( (n1 + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1 );

  double *d_A = NULL;
  cudaMalloc( (void**) &d_A, n * n1 * sizeof(double) );
  cudaMemcpy(d_A, a, n * n1 * sizeof(double), cudaMemcpyHostToDevice); 

  double *d_B = NULL;
  cudaMalloc( (void**) &d_B, n * n1 * sizeof(double) );
  cudaMemcpy(d_B, b, n * n1 * sizeof(double), cudaMemcpyHostToDevice); 

  double *d_C = NULL;
  cudaMalloc( (void**) &d_C, n * n1 * sizeof(double) );

  gpuaddkernel<<< grid, block >>>(d_A, d_B, d_C, n, n1);

  printf("Done with add kernel\n");  

  cudaDeviceSynchronize();

  printf("Done with add kernel and synchronize\n");  

  double *C_gpu = new double[n*n1];
  cudaMemcpy(C_gpu, d_C, n * n1 * sizeof(double), cudaMemcpyDeviceToHost);

  std::memcpy( (void*)c, (void*)C_gpu, n * n1 * sizeof(double) );

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(C_gpu);
}

#endif
