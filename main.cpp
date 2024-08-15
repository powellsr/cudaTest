#include <iostream>
//#include "book.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

#include "gpuintadd.cuh"
#include "gpuadd.cuh"
#include "cpuadd.h"


int main(void) {
  kernel();
  std::cout << "Hello World!\n";
 
  int c;

  gpuintadd( 2, 7, &c );

  std::cout << "2 + 7 = " << c << "\n";
  

  int n = 1<<10; // 20 is 1M elements
  int n1 = 1<<10;
  
  double *x = new double[n];
  double *y = new double[n];
  double *cpuz = new double[n];
  double *mat = new double[n*n1];
  double *mat2 = new double[n*n1];
  double *mat3 = new double[n*n1];
  
  double *a, *b; // d needs to be a pointer on host so we can access it
  double *d = new double[n];
  cudaMalloc( (void**)&a, n*sizeof(double)); // this allocates a pointer on gpu
  cudaMalloc( (void**)&b, n*sizeof(double));
  

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n1; ++j) {
      mat[(j * n1) + i] = 1.0;
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n1; ++j) {
      mat2[(j * n1) + i] = 2.0;
    }
  }

  
  for (int i = 0; i < n; ++i) {
    x[i] = 1.0;
    y[i] = 2.0;
  }

  cudaMemcpy( a, x, n * sizeof(double), cudaMemcpyHostToDevice ); // copying contents of x, y to device vectors a,b (x,y added on CPU)
  cudaMemcpy( b, y, n * sizeof(double), cudaMemcpyHostToDevice ); 

  cpuadd(x, y, cpuz, n);
  gpuadd(a, b, d, n);  
  gpumatadd(mat, mat2, mat3, n, n1);  
  std::cout << "Exited gpumat add\n";

  double maxError = 0.0;
  for (int i = 0; i < n; ++i)
    maxError = fmax(maxError, abs(cpuz[i] - 3.0));
  std::cout << "Max error for cpu vector add: " << maxError << std::endl;  
  
  maxError = 0.0;
  for (int i = 0; i < n; ++i)
    maxError = fmax(maxError, abs(d[i] - 3.0));
  std::cout << "Max error for gpu vector add: " << maxError << std::endl;  

  maxError = 0.0;
  for (int i = 0; i < n * n1; ++i)
    maxError = fmax(maxError, abs(mat3[i] - 3.0));
  std::cout << "Max error for gpu matrix addition: " << maxError << std::endl;  

  delete [] x;
  delete [] y;
  delete [] cpuz;
  delete [] mat;
  delete [] d;
  delete [] mat2;
  delete [] mat3;

  cudaFree(a);
  cudaFree(b);

  return 0;
}

