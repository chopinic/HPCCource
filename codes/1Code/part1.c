#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
void init(double u[firstDim][N], int i_first, int i_last, int rank, int size)
{
  long int t = (long int)time(NULL);
  srand48(t);
  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      // deterministic input
      // int st = (N/size)*rank;
      // u[n1][n2] = (double)1 / ((double)(n1-ml+st) * 1.1 + 1.2 + (double)n2);
      u[n1][n2] = drand48(); // For debugging, make this not random!
    }
  }
}

void sycArray(double arr[firstDim][N], int rank, int size)
{
    if (rank < size - 1){
			// printf("Sending row %d from rank %d to rank %d\n", maxn/size, rank, rank+1);
		    MPI_Send( arr[firstDim-ml*2], N*ml, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD );
		}
		if (rank > 0){
			// printf("Recieving row %d from rank %d to rank %d\n", 0, rank-1, rank);
		    MPI_Recv( arr[0], N*ml, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		}
		if (rank > 0){ 
			// printf("Sending row %d from rank %d to rank %d\n", 1, rank, rank-1);
		    MPI_Send( arr[ml], N*ml, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD );
		}
		if (rank < size - 1){
			// printf("Recieving row %d from rank %d to rank %d\n", maxn/size+1, rank+1, rank);
		    MPI_Recv( arr[firstDim-ml], N*ml, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		}
  MPI_Barrier(MPI_COMM_WORLD);
}

void dudt(double u[firstDim][N], double du[firstDim][N], int i_first, int i_last,int rank, int size)
{
  double sum;
  int count;
  int lowLimit = 0;
  int highLimit = firstDim;
  if (rank==0)
    lowLimit+=ml;
  if (rank==size-1)
    highLimit-=ml;
  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      sum = 0.0;
      count = 0;
      for (int l1 = n1 - ml; l1 <= n1 + ml; l1++)
      {
        for (int l2 = n2 - ml; l2 <= n2 + ml; l2++)
        {
          if ((l1 >= lowLimit) && (l1 < highLimit) && (l2 >= 0) && (l2 < N))
          {
            sum += u[l1][l2]; // Accumulate the local average in sum
            count++;          // Track the count!
          }
        }
      }
      du[n1][n2] =
          u[n1][n2] * (1.0 - sum / count); // And then the actual right-hand-side of the equations
    }
  }
}

void step(double u[firstDim][N], double du[firstDim][N], int i_first, int i_last, int size)
{
  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      u[n1][n2] += h * du[n1][n2];
      if (u[n1][n2] > 1.0)
      {
        u[n1][n2] = 1.0;
      }
      else if (u[n1][n2] < 0.0)
      {
        u[n1][n2] = 0.0;
      }
    }
  }
}

void stat(double stats[2], double u[firstDim][N], int i_first, int i_last)
{
  double mean = 0.0;
  double local_mean = 0.0;
  double var = 0.0;
  double local_var = 0.0;

  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      local_mean += u[n1][n2];
    }
  }

  MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  mean /= (N * N);

  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      local_var += (u[n1][n2] - mean) * (u[n1][n2] - mean);
    }
  }

  MPI_Allreduce(&local_var, &var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  var /= (N * N);
  stats[0] = mean;
  stats[1] = var;
}

void writeFile(double u[firstDim][N], int m) {
  char outstate[80];
  int fileSuccess = sprintf(outstate, "./part1out/state_%i.txt", m);
  if (fileSuccess > 0) {
    FILE *fptr = fopen(outstate, "w");
    for (int n1 = 0; n1 < firstDim; n1++) {
      for (int n2 = 0; n2 < N; n2++) {
        // this segfaults when fptr is null.
        fprintf(fptr, "%2.4f\t", u[n1][n2]);
      }
      fprintf(fptr, "\n");
    }
  } else {
    printf("Failed to write state_%i.txt!\n", m);
  }
};

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i_first = ml;
  int i_last = (N / size)+ml-1;

  double u[firstDim][N];
  double du[firstDim][N];
  double stats[2];
  FILE *fptr;
  if (rank == 0)
  {
    printf("now N: %d", N);

    fptr = fopen("stats.txt", "w");
    fprintf(fptr, "#\tt\tmean\tvar\n");
    printf("#\tt\tmean\tvar\n");
  }
  init(u, i_first, i_last, rank, size);
  sycArray(u,rank,size);
  stat(stats, u, i_first, i_last);

  for (int m = 0; m < M; m++)
  {
    dudt(u, du, i_first, i_last, rank, size);
    sycArray(du,rank,size);
    if (m % mm == 0)
    {
      stat(stats, u, i_first, i_last);
      if (rank == 0)
      {
        fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
        // writeFile(u, m);

        printf("\t%2.2f\t%2.10f\t%2.10f\n", m * h, stats[0], stats[1]);
      }
    }

    step(u, du, i_first, i_last, size);
    sycArray(u,rank,size);
  }

  MPI_Finalize();
  return 0;
}
