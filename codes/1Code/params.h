#pragma once
#define _XOPEN_SOURCE
// weak scaling: 
// 32 1
// 64 4 
// 128 16
// 256 64

const double h = 0.01;
int N = 128;
const int M = 10000;
const int l = 3;
const int ml = 1;
const int mm = 1000;
int addCacheSize = 129; 

void init(double u[N][N], int i_first, int i_last, int world_size);

void step(double u[N][N], double du[N][N], int i_first, int i_last, int world_size);

void dudt(double u[N][N], double du[N][N], int i_first, int i_last, int world_size);

void stat(double stats[2], double u[N][N], int i_first, int i_last);
