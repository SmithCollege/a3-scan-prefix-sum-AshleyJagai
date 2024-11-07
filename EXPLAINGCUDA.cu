//runs in parallel on the GPU, and the main function is the host code
#include <stdio.h>

// NOTES: 
// The host and device in CUDA have separate memory spaces
// (CUDA C kernels can also allocate device memory on devices that support it).
//  gridDim which contains the dimensions of the grid as specified in the first execution configuration parameter to the launch.
// RUN CODE: 1. nvcc -o saxpy saxpy.cu AND 2. ./saxpy
//DEVICE CODE
__global__
//saxpy is the kernel that runs in parallel on the GPU, and the main function is the host code
void saxpy(int n, float a, float *x, float *y)
{
  // blockDim = contains the dimensions of each thread block for the kernel launch. 
  // threadIdx= contains the index of the thread within its thread block 
  // blockIDx = contains the index of the thread block within the grid
  //int i = generates a global index that is used to access elements of the arrays.
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

//HOST CODE
int main(void)
{
  int N = 1<<20;
  //declares two pairs of arrays X and Y
  float *x, *y, *d_x, *d_y;
  //The pointers x and y point to the host arrays
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  //d_x and d_y arrays point to device arrays allocated with the cudaMalloc. LIVES ON THE GPU
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  //Here we set x to an array of ones, and y to an array of twos.
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  // initialize device arrays by coping data from c and Y to d_x and d_y
  // we use cudaMemcpyHostToDevice to specify that the first (destination) argument is a device pointer and the second (source) argument is a host pointer.
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements 
  // launches SAXPY kernel with thread blocks containing 256 threads, and determine the number of thread blocks required to process all N elements of the arrays ((N+255)/256).
  // The first argument in the execution configuration specifies the number of thread blocks in the grid, and the second specifies the number of threads in a thread block.
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  // To get the results back to the host, we copy from the device array pointed to by d_y to the host array pointed to by y by using cudaMemcpy with cudaMemcpyDeviceToHost.
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  //free any allocated memory
  //device memory = cudaMalloc()
  //Host memory = free()
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}