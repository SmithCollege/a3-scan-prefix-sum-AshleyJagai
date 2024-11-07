#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 128

// Single-threaded CPU approach
void single_threaded_scan(int* input, int* output, int size) {
    int running_total = 0;
    for (int i = 0; i < size; i++) {
        running_total += input[i];
        output[i] = running_total;
    }
}

// Get time
double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { 
        printf("Get time of day error\n"); 
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}
