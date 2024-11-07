#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h> 
#include <sys/time.h>

#define SIZE 128


void initInputs(int* input, int size);
void printOutput(int* output, int size);
double get_clock();
__global__ void naive_scan_kernel(int* input, int* output, int size);
__global__ void recursive_doubling_scan_kernel(int* input, int* output, int size);

void initInputs(int* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = i + 1; 
    }
}

void printOutput(int* output, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
}

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { 
        printf("gettimeofday error\n"); 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
    // Allocate memory for singlethread scan in/out (cpu)
    int* stInput = (int*)malloc(sizeof(int) * SIZE);
    int* stOutput = (int*)malloc(sizeof(int) * SIZE);

    // Allocate memory for naive scan in/out (gpu)
    int* nInput;
    int* nOutput;

    // Allocate memory for recursive double scan in/out (gpu)
    int* rInput;
    int* rOutput;

    // Initialize inputs (CPU)
    initInputs(stInput, SIZE);

    //allocate mem (gpu)
    cudaMalloc((void**)&nInput, sizeof(int) * SIZE);   
    cudaMalloc((void**)&nOutput, sizeof(int) * SIZE);
    cudaMallocManaged((void**)&rInput, sizeof(int) * SIZE);
    cudaMallocManaged((void**)&rOutput, sizeof(int) * SIZE);

    // Copy data from CPU to GPU
    cudaMemcpy(nInput, stInput, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(rInput, stInput, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    
    // // Time for scan
    // double start_time = get_clock();
    // single_threaded_scan(stInput, stOutput, SIZE);
    // double end_time = get_clock();

    // // Print results
    // printf("scan Output(CPU):\n");
    // printOutput(stOutput, SIZE);
    // printf("scan Time(CPU): %f seconds\n", end_time - start_time);


    // Time for Naive Scan
    double start_time = get_clock();
    naive_scan_kernel<<<(SIZE + 255) / 256, 256>>>(nInput, nOutput, SIZE);
    cudaDeviceSynchronize(); 
    double end_time = get_clock();

    // Copy results back to CPU
    cudaMemcpy(stOutput, nOutput, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Naive scan Output:\n");
    printOutput(stOutput, SIZE);
    printf("Naive scan Time: %f seconds\n", end_time - start_time);

   // Timing for Recursive Doubling GPU Scan
    start_time = get_clock();
     
    // Run the kernel
    int BLOCK_SIZE = 256; // threads per block
    int blocksPerGrid = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    int threadsPerBlock = BLOCK_SIZE;
    recursive_doubling_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(rInput, rOutput, SIZE);
    cudaDeviceSynchronize(); 
    end_time = get_clock();

    // Copy results back to CPU
    cudaMemcpy(stOutput, rOutput, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Recursive scan Output:\n");
    printOutput(stOutput, SIZE);
    printf("Recursive scan Time: %f seconds\n", end_time - start_time);

    // Clean up memory
    free(stInput);
    free(stOutput);
    cudaFree(nInput);
    cudaFree(nOutput);
    cudaFree(rInput);
    cudaFree(rOutput);

    return 0;
}
