#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>

// CUDA kernel for complex matrix multiplication
__global__ void complexMatrixMultiply(float* A, float* B, float* C, float* D, float* E, float* F, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float e_val = 0.0f;
        float f_val = 0.0f;

        for (int k = 0; k < N; ++k) {
            // Calculate real and imaginary parts
            e_val += A[row * N + k] * C[k * N + col] - B[row * N + k] * D[k * N + col];
            f_val += A[row * N + k] * D[k * N + col] + B[row * N + k] * C[k * N + col];
        }

        E[row * N + col] = e_val;
        F[row * N + col] = f_val;
    }
}

// CPU-based matrix multiplication (for verification)
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

// CUDA wrapper for matrix multiplication
void complexMatrixMultiplication(float* A, float* B, float* C, float* D, float* E, float* F, int N) {
    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    size_t matrixSize = N * N * sizeof(float);

    // Allocate memory on device
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    cudaMalloc((void**)&d_D, matrixSize);
    cudaMalloc((void**)&d_E, matrixSize);
    cudaMalloc((void**)&d_F, matrixSize);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, matrixSize, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel
    complexMatrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D, d_E, d_F, N);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy the result matrices back to host
    cudaMemcpy(E, d_E, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(F, d_F, matrixSize, cudaMemcpyDeviceToHost);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken for matrix multiplication (GPU): %.3f Milliseconds\n", milliseconds);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_F);
}

double get_wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
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

    printf("Performing Complex Matrix Multiplication on GPU (CUDA)...\n");
    // Measure GPU time for matrix multiplication
    complexMatrixMultiplication(A, B, C, D, E_gpu, F_gpu, N);

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
