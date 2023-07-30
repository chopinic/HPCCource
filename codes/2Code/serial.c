#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init(double u[N][N]) {
  long int t = (long int)time(NULL);
  srand48(t);
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      // deterministic input
      // u[n1][n2] =  (double)1/((double)n1*1.1 + 1.2 + (double)n2);
      u[n1][n2] = drand48(); // For debugging, make this not random!
    }
  }
};

void computeIntegralImage(double u[N][N], double addCache[addCacheSize][addCacheSize]) {
  for (int n1 = 1; n1 < addCacheSize; n1++) {
    addCache[n1][0] = 0.0;
    for (int n2 = 1; n2 < addCacheSize; n2++) {
      addCache[n1][n2] = u[n1 - 1][n2 - 1] + addCache[n1 - 1][n2];
    }
  }

  for (int n2 = 1; n2 < addCacheSize; n2++) {
    addCache[0][n2] = 0.0;
    for (int n1 = 1; n1 < addCacheSize; n1++) {
      addCache[n1][n2] += addCache[n1][n2 - 1];
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

        // 0.00    0.01025 0.00019
        // 1.00    0.02664 0.00076
        // 2.00    0.06671 0.00274
        // 3.00    0.15533 0.00776
        // 4.00    0.31883 0.01423
        // 5.00    0.54442 0.01386
        // 6.00    0.75649 0.00651
        // 7.00    0.89195 0.00166
        // 8.00    0.95701 0.00030
void dudt(double u[N][N], double du[N][N], double addCache[addCacheSize][addCacheSize]) {
  computeIntegralImage(u, addCache);

  double mean;
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      mean = computeLocalMean(addCache, n1, n2);
      du[n1][n2] = u[n1][n2] * (1.0 - mean);
    }
  }
}

void step(double u[N][N], double du[N][N]) {
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
};

void stat(double stats[2], double u[N][N]) {
  double mean = 0.0;  
  double var = 0.0;

  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      mean += u[n1][n2];
    }
  }
  mean /= (N * N);
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      var += (u[n1][n2] - mean) * (u[n1][n2] - mean);
    }
  }
  var /= (N * N);

  stats[0] = mean;
  stats[1] = var;
};

void write(double u[N][N], int m) {
  char outstate[80];
  int fileSuccess = sprintf(outstate, "./part2serial/state_%i.txt", m);
  if (fileSuccess > 0) {
    // FILE *fptr = fopen(outstate, "w");
    for (int n1 = 0; n1 < N; n1++) {
      for (int n2 = 0; n2 < N; n2++) {
        // this segfaults when fptr is null.
        // fprintf(fptr, "%2.4f\t", u[n1][n2]);
      }
      // fprintf(fptr, "\n");
    }
  } else {
    printf("Failed to write state_%i.txt!\n", m);
  }
};

int main(int argc, char **argv) {

  double u[N][N];
  double du[N][N];
  double stats[2];
  double addCache[addCacheSize][addCacheSize];
  // FILE *fptr = fopen("./part2serial/stats.txt", "w");
  // fprintf(fptr, "#\tt\tmean\tvar\n");
  printf("#\tt\tmean\tvar\n");

  init(u);
  stat(stats, u);

  for (int m = 0; m < M; m++) {
    dudt(u, du, addCache);
    if (m % mm == 0) {
      stat(stats, u);
      // fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
      // Use these for diagnostic outputs; slow!
      // write(u, m);
      printf("\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
    }
    step(u, du);
  }

  return 0;
};