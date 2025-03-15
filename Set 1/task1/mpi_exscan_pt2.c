#include <mpi.h>
#include <stdio.h>

// Implementation of the Custom MPI_Exscan
void MPI_Exscan_pt2(int rank, int size, int send_val, int *recv_val) {
    int result = 0;

    // Each Process waits to Receive values from Lower Ranks
    for (int i = 0; i < rank; i++) {
        int temp;
        MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result += temp;
    }

    // Each Process Sends values to the Higher Ranks
    for (int i = rank + 1; i < size; i++) {
        MPI_Send(&send_val, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    // Set the final result for the current rank
    *recv_val = (rank != 0) ? result : 0;
}

int main(int argc, char *argv[]) {
    int rank, size, send_val, recv_val;

    // Initialize MPI Communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize Sent values (Simple Process Rank + 1)
    send_val = rank + 1;
    recv_val = 0;

    MPI_Exscan_pt2(rank, size, send_val, &recv_val);

    // Print the Results
    printf("Process %d: exclusive scan result = %d\n", rank, recv_val);

    MPI_Finalize();
    return 0;
}
