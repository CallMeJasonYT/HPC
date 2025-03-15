#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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
                matrix[i * N * N + j * N + k] = (int)(rand_r(&thread_seed) % 1000) / 1000.0;
            }
        }
    }
}

// Validate the binary file data
void validate_data(int *matrix, int *read_matrix, int *exclusive_scan_results, int N, int rank) {
    int error = 0;
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "matrix_data.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // Use OpenMP to handle the data validation for each thread in parallel
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        // Calculate the local offset based on the exclusive scan result for the thread
        MPI_Offset local_offset = (exclusive_scan_results[thread_id]) * N * N * N * sizeof(int);

        // Read the data from the binary file at the calculated offset into the `read_matrix`
        MPI_File_read_at(fh, local_offset, read_matrix, N * N * N, MPI_INT, MPI_STATUS_IGNORE);

        // Check for mismatches between the original matrix and the read matrix
        for (int i = 0; i < N * N * N; i++) {
            if (matrix[i] != read_matrix[i]) {
                error = 1;
                printf("Mismatch at Rank %d, Thread %d: Expected %d, Got %d\n", rank, thread_id, matrix[i], read_matrix[i]);
                break;
            }
        }
        if (!error) {
            printf("Rank %d, Thread %d: Data validation successful.\n", rank, thread_id);
        }
    }

    MPI_File_close(&fh);
}

int main(int argc, char *argv[]) {
    int rank, size, num_threads;
    int N = 10; // Matrix Size
    MPI_File fh;
    MPI_Offset offset;

    // Initialize MPI Communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Spawn OpenMP Threads
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            printf("Process %d: Number of Threads: %d\n", rank, num_threads);
        }
    }

    // Allocate memory for thread values and exclusive scan results and validate it
    int *thread_values = (int *)malloc(num_threads * sizeof(int));
    int *exclusive_scan_results = (int *)malloc(num_threads * sizeof(int));
    if (!thread_values || !exclusive_scan_results) {
        fprintf(stderr, "Memory allocation for MPI_Exscan failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Allocate memory for the matrix and validate it
    int *matrix = (int *)malloc(N * N * N * sizeof(int));
    if (!matrix) {
        fprintf(stderr, "Memory allocation for Matrix failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize the Matrix
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        initialize_matrix(matrix, N, 42, rank, thread_id);
    }

    // Initialize thread values for each thread in the process
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_values[thread_id] = (rank * num_threads) + thread_id + 1;
    }

    MPI_Exscan_omp(rank, size, thread_values, exclusive_scan_results, num_threads);

    // Create matrix_data.bin and Write the Data Into the file with offset calculated based on Custom MPI_Exscan
    MPI_File_open(MPI_COMM_WORLD, "matrix_data.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        MPI_Offset local_offset = (exclusive_scan_results[thread_id]) * N * N * N * sizeof(int);
        MPI_File_write_at(fh, local_offset, matrix, N * N * N, MPI_INT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);

    // Allocate memory for the Read matrix
    int *read_matrix = (int *)malloc(N * N * N * sizeof(int));
    if (!read_matrix) {
        fprintf(stderr, "Memory allocation for read_matrix failed!\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    validate_data(matrix, read_matrix, exclusive_scan_results, N, rank);

    // Clean up
    free(thread_values);
    free(exclusive_scan_results);
    free(matrix);
    free(read_matrix);

    MPI_Finalize();
    return 0;
}
