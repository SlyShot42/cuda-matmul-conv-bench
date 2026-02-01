#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

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

// Exposed C function for Python
extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

__global__ void conv2dGPU(unsigned int *d_A, int *d_k, int *d_B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) {
        int sum = 0;
        int r = N / 2;
        for (int u = 0; u < N; u++) {
            int x = row + u - r;
            for (int v = 0; v < N; v++) {
                int y = col + v - r;
                if (x >= 0 && x < M && y >= 0 && y < M) {
                    sum += d_A[x * M + y] * d_k[u * N + v];
                }
            }
        }
        d_B[row * M + col] = sum;
    }
}

extern "C" void gpu_convolution(unsigned int *h_A, int *h_k, int *h_B, int M, int N) {
    size_t size_A = M * M * sizeof(unsigned int);
    size_t size_k = N * N * sizeof(int);
    size_t size_B = M * M * sizeof(int);

    unsigned int *d_A;
    int *d_k, *d_B;

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_k, size_k);
    cudaMalloc((void **)&d_B, size_B);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    int gridx = (int)ceil((float)M / block.x);
    int gridy = (int)ceil((float)M / block.y);
    dim3 grid(gridx, gridy);

    conv2dGPU<<<grid, block>>>(d_A, d_k, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size_B, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_k);
    cudaFree(d_B);
}