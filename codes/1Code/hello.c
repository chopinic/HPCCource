#include <stdio.h>
#include <mpi.h>

#define LIST_SIZE 100
#define BLOCK_SIZE (LIST_SIZE / 10)

int main(int argc, char** argv) {
    int rank, size;
    int list[LIST_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 10) {
        printf("This program requires exactly 10 processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Initialize the local block of the list for each process
    for (int i = 0; i < BLOCK_SIZE; i++) {
        list[rank * BLOCK_SIZE + i] = 10 * rank + i;
    }

    // Gather all the local blocks into the global list using MPI_Allgather
    MPI_Allgather(MPI_IN_PLACE, BLOCK_SIZE, MPI_INT, list, BLOCK_SIZE, MPI_INT, MPI_COMM_WORLD);

    // Print the list
    if (rank == 0) {
        printf("Merged list:\n");
        for (int i = 0; i < LIST_SIZE; i++) {
            printf("%d ", list[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
