#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>

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

unsigned int* loadPgm(char *path, int *M) {
    FILE *f = fopen(path, "rb");
    if (!f) { return NULL; }

    char format[3];
    fscanf(f, "%2s", format);

    fscanf(f, "%d %d", M, M);

    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f);

    size_t size = (size_t)(*M) * (*M);
    unsigned char *tmp = (unsigned char *)malloc(size);
    fread(tmp, 1, size, f);
    fclose(f);

    unsigned int *A = (unsigned int *)malloc(size * sizeof(unsigned int));
    for (size_t i = 0; i < size; i++) {
        A[i] = tmp[i];
    }

    free(tmp);
    return A;
}


void saveRaw(char *path, int *B, int M) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror("fopen"); exit(1); 
    }
    fwrite(B, sizeof(int), M*M, f);
    fclose(f);
}

double runtimeConv2dGPU(char *path, unsigned int *h_A, int *h_k, int *h_B, int M, int N) {
    size_t sizeA = M * M * sizeof(unsigned int);
    size_t sizek = N * N * sizeof(int);
    size_t sizeB = M * M * sizeof(int);

    unsigned int *d_A;
    int *d_k, *d_B;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_k, sizek);
    cudaMalloc((void **)&d_B, sizeB);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, sizek, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    int gridx = (int)ceil((float)M / block.x);
    int gridy = (int)ceil((float)M / block.y);
    dim3 grid(gridx, gridy);

    clock_t start = clock();
    conv2dGPU<<<grid, block>>>(d_A, d_k, d_B, M, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost);
    saveRaw(path, h_B, M);
    cudaFree(d_A);
    cudaFree(d_k);
    cudaFree(d_B);
    return elapsed;
}


int edgeDetection[9] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

int sharpen5[25] = {
    0, 0,-1, 0, 0,
    0,-1,-2,-1, 0,
    -1,-2,25,-2,-1,
    0,-1,-2,-1, 0,
    0, 0,-1, 0, 0
};


int sharpen7[49] = {
    0, 0, 0,-1, 0, 0, 0,
    0, 0,-1,-2,-1, 0, 0,
    0,-1,-2,-3,-2,-1, 0,
    -1,-2,-3,49,-3,-2,-1,
    0,-1,-2,-3,-2,-1, 0,
    0, 0,-1,-2,-1, 0, 0,
    0, 0, 0,-1, 0, 0, 0
};


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s image.pgm output_dir\n", argv[0]);
        return 1;
    }

    char *inputImagePath = argv[1];
    char *outputDir = argv[2];
    int M;
    unsigned int *A = loadPgm(inputImagePath, &M);
    
    if (!A) {
        printf("Invalid image\n");
        return 1;
    }
    
    size_t size = M * M * sizeof(int);
    int *B = (int *)malloc(size);

    char outputEdgePath[256];
    char outputSharpenPath[256];
    char outputSharpen7Path[256];

    snprintf(outputEdgePath, sizeof(outputEdgePath), "%s/output_edge_gpu.bin", outputDir);
    snprintf(outputSharpenPath, sizeof(outputSharpenPath), "%s/output_sharpen_gpu.bin", outputDir);
    snprintf(outputSharpen7Path, sizeof(outputSharpen7Path), "%s/output_sharpen7_gpu.bin", outputDir);

    double elapsed = runtimeConv2dGPU(outputEdgePath, A, edgeDetection, B, M, 3);
    printf("Edge Detection GPU execution time (M=%d, N=%d): %f seconds\n", M, 3, elapsed);

    elapsed = runtimeConv2dGPU(outputSharpenPath, A, sharpen5, B, M, 5);
    printf("Sharpen GPU execution time (M=%d, N=%d): %f seconds\n", M, 5, elapsed);

    elapsed = runtimeConv2dGPU(outputSharpen7Path, A, sharpen7, B, M, 7);
    printf("Sharpen7 GPU execution time (M=%d, N=%d): %f seconds\n", M, 7, elapsed);

    free(A);
    free(B);

    return 0;
}