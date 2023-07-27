#include "params.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init(double u[N][N]) {
  long int t = (long int)time(NULL);
  // srand48(t);
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      u[n1][n2] =  (double)1/((double)n1*1.1 + 1.2 + (double)n2);
      // u[n1][n2] = drand48(); // For debugging, make this not random!
    }
  }
};

void dudt(double u[N][N], double du[N][N]) {
  double sum;
  int count;
  for (int n1 = 0; n1 < N; n1++) {
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
          u[n1][n2] * (1.0 - sum / count); // And then the actual
                                           // right-hand-side of the equations
    }
  }
};

void step(double u[N][N], double du[N][N]) {
  for (int n1 = 0; n1 < N; n1++) { 
    for (int n2 = 0; n2 < N; n2++) {
      u[n1][n2] += h * du[n1][n2];
    }
  }
};

void stat(double stats[2], double u[N][N]) {
  double mean = 0.0;
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      mean += u[n1][n2] / (N * N);
    }
  }
  stats[0] = mean;
  double var = 0.0;
  for (int n1 = 0; n1 < N; n1++) {
    for (int n2 = 0; n2 < N; n2++) {
      var += (u[n1][n2] - mean) * (u[n1][n2] - mean) / (N * N);
    }
  }
  stats[1] = var;
};

void write(double u[N][N], int m) {
  char outstate[80];
  int fileSuccess = sprintf(outstate, "./part2serial/state_%i.txt", m);
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
};

int main(int argc, char **argv) {

  double u[N][N];
  double du[N][N];
  double stats[2];

  FILE *fptr = fopen("./part2serial/stats.txt", "w");
  fprintf(fptr, "#\tt\tmean\tvar\n");
  printf("#\tt\tmean\tvar\n");

  init(u);
  stat(stats, u);

  for (int m = 0; m < M; m++) {
    dudt(u, du);
    if (m % mm == 0) {
      stat(stats, u);
      fprintf(fptr, "\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
      // Use these for diagnostic outputs; slow!
      write(u, m);
      printf("\t%2.2f\t%2.5f\t%2.5f\n", m * h, stats[0], stats[1]);
    }
    step(u, du);
  }

  return 0;
};