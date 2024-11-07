#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

__global__ void naive_scan_kernel(int* input, int* output, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        int value = 0;
        for (int j = 0; j <= i; j++) {
            value += input[j];
        }
        output[i] = value;
    }
}
