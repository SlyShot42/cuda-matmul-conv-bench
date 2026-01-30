#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) { 
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; 

    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 

    int Row = by * TILE_WIDTH + ty; 
    int Col = bx * TILE_WIDTH + tx; 
    
    float Pvalue = 0.0; 
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { 
        if (Row < N && (m*TILE_WIDTH+tx) < N) {
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        } 
        else {
            ds_A[ty][tx] = 0.0f; 
        }

        if (Col < N && (m*TILE_WIDTH+ty) < N) {
            ds_B[ty][tx] = B[(m*TILE_WIDTH + ty) * N + Col];
        } 
        else {
            ds_B[ty][tx] = 0.0f; 
        }
        __syncthreads(); 
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        } 
        __syncthreads(); 
    } 
    
    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
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

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    int gridx = (int)ceil((float)N / block.x);
    int gridy = (int)ceil((float)N / block.y);
    dim3 grid(gridx, gridy);

    clock_t start = clock();
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
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

