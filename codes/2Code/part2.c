#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void init(double u[N][N]) {
  long int t = (long int)time(NULL);
  srand48(t);
  #pragma omp parallel for collapse(2)
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      // deterministic input
      // u[n1][n2] =  (double)1/((double)n1*1.1 + 1.2 + (double)n2);
      u[n1][n2] = drand48(); // For debugging, make this not random!

    }
  }
}


void computeIntegralImage(double u[N][N], double addCache[addCacheSize][addCacheSize]) {
  #pragma omp parallel for
  for (int n1 = 1; n1 < addCacheSize; n1++) {
    addCache[n1][0] = 0.0;
    for (int n2 = 1; n2 < addCacheSize; n2++) {
      addCache[n1][n2] = u[n1 - 1][n2 - 1] + addCache[n1][n2-1];
    }
  }
  #pragma omp parallel for
  for (int n2 = 1; n2 < addCacheSize; n2++) {
    addCache[0][n2] = 0.0;
    for (int n1 = 1; n1 < addCacheSize; n1++) {
      addCache[n1][n2] += addCache[n1-1][n2];
    }
  }
}

double computeLocalMean(double addCache[addCacheSize][addCacheSize], int n1, int n2) {
  int l1_start = n1 - ml;
  if (l1_start < 0)
    l1_start = 0;
  int l1_end = n1 + ml + 1;
  if (l1_end > N)
    l1_end = N;
  int l2_start = n2 - ml;
  if (l2_start<0)
    l2_start = 0;
  int l2_end = n2 + ml + 1;
  if (l2_end>N)
    l2_end=N;

  int count_total = (l1_end - l1_start) * (l2_end - l2_start);

  double sum = addCache[l1_end][l2_end] - addCache[l1_end][l2_start] -
               addCache[l1_start][l2_end] + addCache[l1_start][l2_start];

  double mean = sum / count_total;
  return mean;
}

void dudt(double u[N][N], double du[N][N], double addCache[addCacheSize][addCacheSize]) {
  computeIntegralImage(u, addCache);

  double mean;
  #pragma omp parallel for collapse(2) private(mean)
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      mean = computeLocalMean(addCache, n1, n2);
      du[n1][n2] = u[n1][n2] * (1.0 - mean);
    }
  }
}

// void dudt(double u[N][N], double du[N][N]) {
//   double sum;
//   int count;
//   #pragma omp parallel for collapse(2) private(sum, count)
//   for (int n1 = 0; n1 < N; n1++) {
//     for (int n2 = 0; n2 < N; n2++) {
//       sum = 0.0;
//       count = 0;
//       for (int l1 = n1 - ml; l1 <= n1 + ml; l1++) {
//         for (int l2 = n2 - ml; l2 <= n2 + ml; l2++) {
//           if ((l1 >= 0) && (l1 < N) && (l2 >= 0) && (l2 < N)) {
//             sum += u[l1][l2]; // Accumulate the local average in sum
//             count++;          // Track the count!
//           }
//         }
//       }
//       du[n1][n2] =
//           u[n1][n2] * (1.0 - sum / count); // And then the actual
//                                            // right-hand-side of the equations
//     }
//   }
// }

void step(double u[N][N], double du[N][N]) {
  #pragma omp parallel for collapse(2)
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      u[n1][n2] += h * du[n1][n2];
      if (u[n1][n2] > 1.0) {
        u[n1][n2] = 1.0;
      } else if (u[n1][n2] < 0.0) {
        u[n1][n2] = 0.0;
      }
    }
  }
}

void stat(double stats[2], double u[N][N]) {
  double mean = 0.0;
  double var = 0.0;

  #pragma omp parallel for collapse(2) reduction(+:mean)
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      mean += u[n1][n2];
    }
  }
  mean /= (N * N);

  #pragma omp parallel for collapse(2) reduction(+:var)
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      var += (u[n1][n2] - mean) * (u[n1][n2] - mean);
    }
  }
  var /= (N * N);

  stats[0] = mean;
  stats[1] = var;
}

int main(int argc, char **argv) {
  if (argc >= 3) {
      for (int i = 1; i < argc; i++) {
          if (strcmp(argv[i], "-NN") == 0) {
              if (i + 1 < argc) {
                  N = atoi(argv[i + 1]);
              }
          }
      }
  }
  addCacheSize = N + 1;
  double u[N][N];
  double du[N][N];
  double stats[2];
  double addCache[addCacheSize][addCacheSize];

  // FILE *fptr = fopen("./part2out/stats.txt", "w");
  // fprintf(fptr, "#\tt\tmean\tvar\n");
  printf("now N: %d",N);
  printf("#\tt\tmean\tvar\n");

  init(u);
  stat(stats, u);

  for (int m = 0; m < M; m++) {
    dudt(u, du, addCache);
    if (m % mm == 0) {
      stat(stats, u);
      // fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
      printf("\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
    }
    step(u, du);
  }

  // fclose(fptr); // Close the stats.txt file after writing all statistics
  
  return 0;
}
