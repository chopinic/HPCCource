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
const int mm = 100;
int firstDim = 128 / 4 + 1 * 2;

void init(double u[firstDim][N], int i_first, int i_last, int rank, int size);

void step(double u[firstDim][N], double du[firstDim][N], int i_first, int i_last, int size);

void dudt(double u[firstDim][N], double du[firstDim][N], int i_first, int i_last,int rank, int size);

void stat(double stats[2], double u[firstDim][N], int i_first, int i_last);

void sycArray(double arr[firstDim][N], int rank, int size);