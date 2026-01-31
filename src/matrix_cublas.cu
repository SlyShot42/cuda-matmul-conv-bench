#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaDeviceSynchronize();
    clock_t start = clock();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cublasDestroy(handle);

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU execution time with cuBLAS (N=%d): %f seconds\n", N, elapsed);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    if (argc > 2 && strcmp(argv[2], "verbose") == 0) {
        printf("C[0]=%f C[mid]=%f C[last]=%f\n", C[0], C[(N/2)*N + (N/2)], C[N*N-1]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
    return 0;
}
