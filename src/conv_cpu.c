#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void conv2dCPU(unsigned int *A, int *k, int *B, int M, int N) {
	int r = N/2;
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < M; j++) {
			int sum = 0;
			for(int u = 0; u < N; u++) {
				int x = i + u - r;
				for(int v = 0; v < N; v++) {
					int y = j + v - r;
					if(x >= 0 && y >= 0 && x < M && y < M) {
						sum += A[x * M + y] * k[u * N + v];
					}
				}
			}
			B[i * M + j] = sum;
		}
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
    unsigned char *tmp = malloc(size);
    fread(tmp, 1, size, f);
    fclose(f);

    unsigned int *A = malloc(size * sizeof(unsigned int));
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

double runtimeConv2dCPU(char *path,unsigned int *A, int *k, int *B, int M, int N) {
    clock_t start = clock();
    conv2dCPU(A, k, B, M, N);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    saveRaw(path, B, M);
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

    snprintf(outputEdgePath, sizeof(outputEdgePath), "%s/output_edge_cpu.bin", outputDir);
    snprintf(outputSharpenPath, sizeof(outputSharpenPath), "%s/output_sharpen_cpu.bin", outputDir);
    snprintf(outputSharpen7Path, sizeof(outputSharpen7Path), "%s/output_sharpen7_cpu.bin", outputDir);

    double elapsed = runtimeConv2dCPU(outputEdgePath, A, edgeDetection, B, M, 3);
    printf("Edge Detection CPU execution time (M=%d, N=%d): %f seconds\n", M, 3, elapsed);

    elapsed = runtimeConv2dCPU(outputSharpenPath, A, sharpen5, B, M, 5);
    printf("Sharpen CPU execution time (M=%d, N=%d): %f seconds\n", M, 5, elapsed);

    elapsed = runtimeConv2dCPU(outputSharpen7Path, A, sharpen7, B, M, 7);
    printf("Sharpen7 CPU execution time (M=%d, N=%d): %f seconds\n", M, 7, elapsed);

    free(A);
    free(B);

    return 0;
}