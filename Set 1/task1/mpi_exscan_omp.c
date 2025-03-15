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

int main(int argc, char *argv[]) {
    int rank, size, num_threads;

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
            printf("Number of Threads: %d\n", num_threads);
        }
    }

    // Allocate memory for thread values and exclusive scan results
    int *thread_values = (int *)malloc(num_threads * sizeof(int));
    int *exclusive_scan_results = (int *)malloc(num_threads * sizeof(int));

    // Initialize thread values for each thread in the process
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        thread_values[thread_id] = (rank * num_threads) + thread_id + 1;
    }

    MPI_Exscan_omp(rank, size, thread_values, exclusive_scan_results, num_threads);

    // Print the Results
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Process %d, Thread %d: exclusive scan result = %d\n",
               rank, thread_id, exclusive_scan_results[thread_id]);
    }

    // Free the Memory
    free(thread_values);
    free(exclusive_scan_results);

    MPI_Finalize();
    return 0;
}
