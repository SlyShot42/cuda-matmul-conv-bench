#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024; // allow matrix size as input
    size_t size = N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    int gridx = (int)ceil((float)N / block.x);
    int gridy = (int)ceil((float)N / block.y);
    dim3 grid(gridx, gridy);

    clock_t start = clock();
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    if (argc > 2 && strcmp(argv[2], "verbose") == 0) {
        printf("C[0]=%f C[mid]=%f C[last]=%f\n", C[0], C[(N/2)*N + (N/2)], C[N*N-1]);
    }

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU execution time (N=%d): %f seconds\n", N, elapsed);
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}