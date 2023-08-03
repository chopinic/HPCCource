#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

void init(double u[N][N], int i_first, int i_last, int world_size)
{
  long int t = (long int)time(NULL);
  srand48(t);
  for (int n1 = i_first; n1 <= i_last; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      // deterministic input
      u[n1][n2] = (double)1 / ((double)n1 * 1.1 + 1.2 + (double)n2);
      // u[n1][n2] = drand48(); // For debugging, make this not random!
    }
  }
  MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, u, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void tranpose(double addCache[addCacheSize][addCacheSize])
{
  for (int i = 0; i < addCacheSize; i++)
  {
    for (int ii = 0; ii < addCacheSize; ii++)
    {
      double t = addCache[i][ii];
      addCache[i][ii] = addCache[ii][i];
      addCache[ii][i] = t;
    }
  }
}

void computeIntegralImage(double u[N][N], int i_first, int i_last, double addCache[addCacheSize][addCacheSize])
{
  double localColumnSum[i_last - i_first + 1][addCacheSize];
  double localRowSum[i_last - i_first + 1][addCacheSize];

  for (int i = 0; i < i_last - i_first; i++)
  {
    localColumnSum[i][0] = 0.0;
    for (int j = 0; j < N; j++)
    {
      localColumnSum[i + 1][j + 1] = u[i+i_first][j] + localColumnSum[i+1][j];
    }
  }
  MPI_Allgather(localColumnSum[1], (i_last - i_first) * addCacheSize, MPI_DOUBLE,
                addCache[1], (i_last - i_first) * addCacheSize, MPI_DOUBLE, MPI_COMM_WORLD);
  if (i_first == 0)
    tranpose(addCache);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < i_last - i_first; i++)
  {
    localRowSum[i][0] = 0.0;
    for (int j = 0; j < N; j++)
    {
      localRowSum[i + 1][j + 1] = addCache[i+1+i_first][j+1] + localRowSum[i+1][j];
    }
  }
  MPI_Allgather(localRowSum[1], (i_last - i_first) * addCacheSize, MPI_DOUBLE,
                addCache[1], (i_last - i_first) * addCacheSize, MPI_DOUBLE, MPI_COMM_WORLD);
  if (i_first == 0)
    tranpose(addCache);
  MPI_Barrier(MPI_COMM_WORLD);
}

double computeLocalMean(double addCache[addCacheSize][addCacheSize], int n1, int n2)
{
  int l1_start = n1 - ml;
  if (l1_start < 0)
    l1_start = 0;
  int l1_end = n1 + ml + 1;
  if (l1_end > N)
    l1_end = N;
  int l2_start = n2 - ml;
  if (l2_start < 0)
    l2_start = 0;
  int l2_end = n2 + ml + 1;
  if (l2_end > N)
    l2_end = N;

  int count_total = (l1_end - l1_start) * (l2_end - l2_start);

  double sum = addCache[l1_end][l2_end] - addCache[l1_end][l2_start] -
               addCache[l1_start][l2_end] + addCache[l1_start][l2_start];

  double mean = sum / count_total;
  return mean;
}

void dudt(double u[N][N], double du[N][N], int i_first, int i_last, int world_size, double addCache[addCacheSize][addCacheSize])
{
  computeIntegralImage(u, i_first, i_last, addCache);

  double mean;
  for (int n1 = 0; n1 < N; n1++)
  {
    for (int n2 = 0; n2 < N; n2++)
    {
      mean = computeLocalMean(addCache, n1, n2);
      du[n1][n2] = u[n1][n2] * (1.0 - mean);
    }
  }
  MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, du, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void step(double u[N][N], double du[N][N], int i_first, int i_last, int world_size)
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
  MPI_Allgather(MPI_IN_PLACE, N * N / world_size, MPI_DOUBLE, u, N * N / world_size, MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

void stat(double stats[2], double u[N][N], int i_first, int i_last)
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

int main(int argc, char **argv)
{
  if (argc >= 3)
  {
    for (int i = 1; i < argc; i++)
    {
      if (strcmp(argv[i], "-NN") == 0)
      {
        if (i + 1 < argc)
        {
          N = atoi(argv[i + 1]);
        }
      }
    }
  }
  addCacheSize = N + 1;
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int i_first = (N / world_size) * world_rank;
  int i_last = i_first + (N / world_size) - 1;

  if (world_rank == world_size - 1)
  {
    i_last = N - 1;
  }

  double u[N][N];
  double du[N][N];
  double stats[2];
  double addCache[addCacheSize][addCacheSize];
  // FILE *fptr;
  if (world_rank == 0)
  {
    printf("now N: %d", N);

    // fptr = fopen("./part1out/stats.txt", "w");
    // fprintf(fptr, "#\tt\tmean\tvar\n");
    printf("#\tt\tmean\tvar\n");
  }
  init(u, i_first, i_last, world_size);
  stat(stats, u, i_first, i_last);

  for (int m = 0; m < M; m++)
  {
    dudt(u, du, i_first, i_last, world_size, addCache);
    if (m % mm == 0)
    {
      stat(stats, u, i_first, i_last);
      if (world_rank == 0)
      {
        // fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
        printf("\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
      }
    }

    step(u, du, i_first, i_last, world_size);
  }

  MPI_Finalize();
  return 0;
}
