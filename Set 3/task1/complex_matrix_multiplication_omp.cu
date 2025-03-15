#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>

// Function to perform complex matrix multiplication on the CPU (for verification)
void cpuComplexMatrixMultiply(float* A, float* B, float* C, float* D, float* E, float* F, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            E[i * N + j] = 0.0f;
            F[i * N + j] = 0.0f;
            for (int k = 0; k < N; ++k) {
                E[i * N + j] += A[i * N + k] * C[k * N + j] - B[i * N + k] * D[k * N + j];
                F[i * N + j] += A[i * N + k] * D[k * N + j] + B[i * N + k] * C[k * N + j];
            }
        }
    }
}

// Function to verify the results of CPU and GPU matrix multiplications
bool verifyResults(float* E_cpu, float* F_cpu, float* E_gpu, float* F_gpu, int N, float tolerance = 1e-6) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(E_cpu[i * N + j] - E_gpu[i * N + j]) > tolerance ||
                fabs(F_cpu[i * N + j] - F_gpu[i * N + j]) > tolerance) {
                printf("Mismatch at (%d, %d)\n", i, j);
                return false;
            }
        }
    }
    return true;
}

// OpenMP-based GPU offloading for complex matrix multiplication
void openmpComplexMatrixMultiply(float* A, float* B, float* C, float* D, float* E, float* F, int N) {
    #pragma omp target teams distribute parallel for collapse(2) map(to: A[0:N*N], B[0:N*N], C[0:N*N], D[0:N*N]) map(from: E[0:N*N], F[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float e_val = 0.0f;
            float f_val = 0.0f;
            for (int k = 0; k < N; ++k) {
                e_val += A[i * N + k] * C[k * N + j] - B[i * N + k] * D[k * N + j];
                f_val += A[i * N + k] * D[k * N + j] + B[i * N + k] * C[k * N + j];
            }
            E[i * N + j] = e_val;
            F[i * N + j] = f_val;
        }
    }
}

double get_wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]); // Size of the matrix

    printf("Matrix Size: %d\n", N);
    
    // Allocate host memory
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float *D = (float*)malloc(N * N * sizeof(float));
    float *E_cpu = (float*)malloc(N * N * sizeof(float)); // Real part of the result (CPU)
    float *F_cpu = (float*)malloc(N * N * sizeof(float)); // Imaginary part of the result (CPU)
    float *E_gpu = (float*)malloc(N * N * sizeof(float)); // Real part of the result (GPU)
    float *F_gpu = (float*)malloc(N * N * sizeof(float)); // Imaginary part of the result (GPU)

    // Initialize matrices with random values
    for (int i = 0; i < N * N; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = rand() % 10;
        D[i] = rand() % 10;
    }

    printf("Performing Complex Matrix Multiplication on CPU (Sequential)...\n");

    // Measure CPU time for matrix multiplication
    double start_time = get_wtime();
    cpuComplexMatrixMultiply(A, B, C, D, E_cpu, F_cpu, N);
    double end_time = get_wtime();
    printf("Time taken for matrix multiplication (CPU): %.3f Seconds\n", end_time - start_time);

    printf("Performing Complex Matrix Multiplication on GPU (OpenMP)...\n");

    // Measure GPU time for matrix multiplication
    start_time = get_wtime();
    openmpComplexMatrixMultiply(A, B, C, D, E_gpu, F_gpu, N);
    end_time = get_wtime();
    printf("Time taken for matrix multiplication (GPU): %.3f Seconds\n", end_time - start_time);

    // Verify results
    if (verifyResults(E_cpu, F_cpu, E_gpu, F_gpu, N)) {
        printf("Verification passed! CPU and GPU results are identical.\n");
    } else {
        printf("Verification failed! There are mismatches between CPU and GPU results.\n");
    }

    // Cleanup
    free(A);
    free(B);
    free(C);
    free(D);
    free(E_cpu);
    free(F_cpu);
    free(E_gpu);
    free(F_gpu);

    return 0;
}
