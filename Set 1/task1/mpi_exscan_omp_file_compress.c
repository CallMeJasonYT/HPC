#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>

// Implementation of Custom MPI Exscan With Hybrid Programming Model
void MPI_Exscan_omp(int rank, int size, int *thread_values, int *exclusive_scan_results, int num_threads) {
    int total_thread_sum = 0;

    // Compute the total sum of thread values using OpenMP Reduce with Sum Operator
    #pragma omp parallel for reduction(+:total_thread_sum)
    for (int t = 0; t < num_threads; t++) {
        total_thread_sum += thread_values[t];
    }

    // Set Received prefix sum to 0 (Will be updated via MPI communication)
    int received_prefix_sum = 0;

    // If the rank is not 0, receive the prefix sum from the previous process
    if (rank != 0) {
        MPI_Recv(&received_prefix_sum, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Store the prefix sum received from the previous process
    int process_prefix_sum = received_prefix_sum;

    // Add the current process's total thread sum to the received prefix sum
    received_prefix_sum += total_thread_sum;

    // If this is not the last process, send the updated prefix sum to the next process
    if (rank < size - 1) {
        MPI_Send(&received_prefix_sum, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Use OpenMP to calculate the exclusive scan for each thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int thread_cumulative_sum = 0;

        // Calculate the cumulative sum for each thread before the current thread
        for (int t = 0; t < thread_id; t++) {
            thread_cumulative_sum += thread_values[t];
        }
        
        // Store the result of the exclusive scan for the current thread
        exclusive_scan_results[thread_id] = process_prefix_sum + thread_cumulative_sum;
    }
}

// Initialize matrix with random values
void initialize_matrix(int *matrix, int N, int seed, int rank, int thread_id) {
    unsigned int thread_seed = seed + rank * 100 + thread_id;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                matrix[i * N * N + j * N + k] = rand_r(&thread_seed) % 100;
            }
        }
    }
}

// Compress matrix
void compress_matrix(const int *matrix, int matrix_size, unsigned char **compressed_data, uLongf *compressed_size) {
    uLongf max_compressed_size = compressBound(matrix_size * sizeof(int));
    *compressed_data = (unsigned char *)malloc(max_compressed_size);
    if (!(*compressed_data)) {
        fprintf(stderr, "Memory allocation for compression failed!\n");
        exit(EXIT_FAILURE);
    }
    if (compress(*compressed_data, &max_compressed_size, (const unsigned char *)matrix, matrix_size * sizeof(int)) != Z_OK) {
        fprintf(stderr, "Compression failed!\n");
        exit(EXIT_FAILURE);
    }
    *compressed_size = max_compressed_size;
}

// Decompress data
void decompress_data(const unsigned char *compressed_data, uLongf compressed_size, int *matrix, int matrix_size) {
    uLongf decompressed_size = matrix_size * sizeof(int);
    if (uncompress((unsigned char *)matrix, &decompressed_size, compressed_data, compressed_size) != Z_OK) {
        fprintf(stderr, "Decompression failed!\n");
        exit(EXIT_FAILURE);
    }
}

// Validate matrix
void validate_matrix(const int *original_matrix, const int *read_matrix, int matrix_size, int rank) {
    int validation_failed = 0;
    #pragma omp parallel for reduction(||:validation_failed)
    for (int i = 0; i < matrix_size; i++) {
        if (original_matrix[i] != read_matrix[i]) {
            validation_failed = 1;
        }
    }
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (!validation_failed) {
            printf("Validation succeeded on rank %d, thread %d.\n", rank, thread_id);
        } else {
            printf("Validation failed on rank %d, thread %d.\n", rank, thread_id);
        }
    }
    if (validation_failed) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int rank, size, num_threads;
    int N = 10; // Matrix dimensions
    MPI_File fh;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            printf("Process %d: Number of Threads: %d\n", rank, num_threads);
        }
    }

    // Allocate memory for thread values and exclusive scan results
    int *thread_values = (int *)malloc(num_threads * sizeof(int));
    int *exclusive_scan_results = (int *)malloc(num_threads * sizeof(int));
    if (!thread_values || !exclusive_scan_results) {
        fprintf(stderr, "Memory allocation for MPI_Exscan failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Allocate memory for the matrix
    int *matrix = (int *)malloc(N * N * N * sizeof(int));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize matrix
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        initialize_matrix(matrix, N, 42, rank, thread_id);
    }

    // Compress matrix
    unsigned char *compressed_data = NULL;
    uLongf compressed_size = 0;
    compress_matrix(matrix, N * N * N, &compressed_data, &compressed_size);

    // Populate thread_values with compressed sizes per thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_values[thread_id] = compressed_size / num_threads; // Example assignment
    }

    MPI_Exscan_omp(rank, size, thread_values, exclusive_scan_results, num_threads);

    // Create matrix_data_compressed.bin and write data with offsets
    MPI_File_open(MPI_COMM_WORLD, "matrix_data_compressed.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MPI_Offset local_offset = (exclusive_scan_results[thread_id]); // Correct offset calculation
        MPI_File_write_at(fh, local_offset, compressed_data, compressed_size, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);

    // Read and validate data
    MPI_File_open(MPI_COMM_WORLD, "matrix_data_compressed.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    unsigned char *read_compressed_data = (unsigned char *)malloc(compressed_size);
    int *read_matrix = (int *)malloc(N * N * N * sizeof(int));

    if (!read_compressed_data || !read_matrix) {
        fprintf(stderr, "Memory allocation for validation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MPI_Offset local_offset = exclusive_scan_results[thread_id];
        MPI_File_read_at(fh, local_offset, read_compressed_data, compressed_size, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    decompress_data(read_compressed_data, compressed_size, read_matrix, N * N * N);

    // Validate the matrix
    validate_matrix(matrix, read_matrix, N * N * N, rank);

    // Clean up
    MPI_File_close(&fh);
    free(matrix);
    free(compressed_data);
    free(read_compressed_data);
    free(read_matrix);
    free(thread_values);
    free(exclusive_scan_results);

    MPI_Finalize();
    return 0;
}