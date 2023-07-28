#pragma once
#define _XOPEN_SOURCE

const double h = 0.01;
const int N = 512;
const int M = 10000;
const int l = 3;
const int ml = (l - 1) / 2;
const int mm = 1000;

void init(double u[N][N]);

void step(double u[N][N], double du[N][N]);

void dudt(double u[N][N], double du[N][N]);

void stat(double states[2], double u[N][N]);