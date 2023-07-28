#!/bin/bash


# Request resources:
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -c 4		# number of CPU cores, one per thread, up to 128
#SBATCH --mem=1G	# memory required, up to 250G on standard nodes
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)
module load gcc

rm ./*.txt
gcc -O3 serial.c -Wall -o serial
time ./serial