#pragma once
#define _XOPEN_SOURCE

const double h = 0.01;
// weak scaling: 
// 32 1
// 64 4 
// 128 16
// 192 36
// 256 64
// 288 81

const int N = 64;
const int M = 10000;
const int l = 3;
const int ml = (l - 1) / 2;
const int mm = 1000;

void init(double u[N][N], int i_first, int i_last, int world_size);

void step(double u[N][N], double du[N][N], int i_first, int i_last, int world_size);

void dudt(double u[N][N], double du[N][N], int i_first, int i_last, int world_size);

void stat(double stats[2], double u[N][N], int i_first, int i_last);
