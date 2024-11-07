#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>


__global__ void recursive_doubling_scan_kernel(int *in, int *out, int SIZE) {
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tIdx < SIZE) {
        out[tIdx] = in[tIdx]; 
    }
    __syncthreads(); 

    // recursive doubling part
    for (int offset = 1; offset < SIZE; offset *= 2) {
        if (tIdx >= offset) {
            out[tIdx] += out[tIdx - offset]; // Accumulate values
        }
        __syncthreads(); 
    }
}