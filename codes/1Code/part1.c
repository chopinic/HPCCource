#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void init(double u[N][N], int i_first, int i_last, int world_size) {
    long int t = (long int)time(NULL);
    srand48(t);
    for (int n1 = i_first; n1 <= i_last; n1++) {
        for (int n2 = 0; n2 < N; n2++) {
            // deterministic input
            // u[n1][n2] = (double)1 / ((double)n1 * 1.1 + 1.2 + (double)n2);
            u[n1][n2] = drand48(); // For debugging, make this not random!
        }
    }
    MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, u, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void dudt(double u[N][N], double du[N][N], int i_first, int i_last, int world_size) {
    double sum;
    int count;
    for (int n1 = i_first; n1 <= i_last; n1++) {
        for (int n2 = 0; n2 < N; n2++) {
            sum = 0.0;
            count = 0;
            for (int l1 = n1 - ml; l1 <= n1 + ml; l1++) {
                for (int l2 = n2 - ml; l2 <= n2 + ml; l2++) {
                    if ((l1 >= 0) && (l1 < N) && (l2 >= 0) && (l2 < N)) {
                        sum += u[l1][l2]; // Accumulate the local average in sum
                        count++;          // Track the count!
                    }
                }
            }
            du[n1][n2] =
                u[n1][n2] * (1.0 - sum / count); // And then the actual right-hand-side of the equations
        }
    }
    MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, du, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void step(double u[N][N], double du[N][N], int i_first, int i_last, int world_size) {
    for (int n1 = i_first; n1 <= i_last; n1++) {
        for (int n2 = 0; n2 < N; n2++) {
            u[n1][n2] += h * du[n1][n2];
        }
    }
    MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, u, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void stat(double stats[2], double u[N][N], int i_first, int i_last) {
    double mean = 0.0;
    double local_mean = 0.0;
    double var = 0.0;
    double local_var = 0.0;

    for (int n1 = i_first; n1 <= i_last; n1++) {
        for (int n2 = 0; n2 < N; n2++) {
            local_mean += u[n1][n2];
        }
    }

    MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mean /= (N * N);

    for (int n1 = i_first; n1 <= i_last; n1++) {
        for (int n2 = 0; n2 < N; n2++) {
            local_var += (u[n1][n2] - mean) * (u[n1][n2] - mean);
        }
    }

    MPI_Allreduce(&local_var, &var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    var /= (N * N);

    stats[0] = mean;
    stats[1] = var;

}

// change func name to writeFile to avoid name conflict with default "write" func in c lib.
void writeFile(double u[N][N], int m) {
    char outstate[80];
    int fileSuccess = sprintf(outstate, "./part1out/state_%i.txt", m);
    if (fileSuccess > 0) {
        FILE *fptr = fopen(outstate, "w");
        for (int n1 = 0; n1 < N; n1++) {
            for (int n2 = 0; n2 < N; n2++) {
                // this segfaults when fptr is null.
                fprintf(fptr, "%2.4f\t", u[n1][n2]);
            }
            fprintf(fptr, "\n");
        }
    } else {
        printf("Failed to write state_%i.txt!\n", m);
    }
}


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int world_rank,world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int i_first = (N / world_size) * world_rank;
    int i_last = i_first + (N / world_size) - 1;

    if (world_rank == world_size - 1) {
        i_last = N - 1;
    }

    double u[N][N];
    double du[N][N];
    double stats[2];

    FILE *fptr;
    if (world_rank == 0) {
        fptr = fopen("./part1out/stats.txt", "w");
        fprintf(fptr, "#\tt\tmean\tvar\n");
        printf("#\tt\tmean\tvar\n");
    }
    init(u, i_first, i_last, world_size);
    stat(stats, u, i_first, i_last);

    for (int m = 0; m < M; m++) {
        dudt(u, du, i_first, i_last, world_size);
        if (m % mm == 0) {
            stat(stats, u, i_first, i_last);
            if (world_rank == 0) {
                fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
                // writeFile(u, m);
                printf("\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
            }
        }

        step(u, du, i_first, i_last, world_size);
    }

    MPI_Finalize();
    return 0;
}
