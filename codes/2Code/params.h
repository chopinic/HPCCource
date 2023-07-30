#pragma once
#define _XOPEN_SOURCE

const double h = 0.01;
int N = 128;
const int M = 1005;
const int l = 3;
const int ml = 1;
const int mm = 100;
int addCacheSize = 129; 

void init(double u[N][N]);

void step(double u[N][N], double du[N][N]);

void dudt(double u[N][N], double du[N][N], double addCache[addCacheSize][addCacheSize]);
// void dudt(double u[N][N], double du[N][N]);

void stat(double states[2], double u[N][N]);