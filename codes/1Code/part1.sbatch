#!/bin/bash

# Request resources:
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -n 65		# number of MPI ranks (1 per CPU core)
#SBATCH --mem=20G	# memory required per node, in units M, G or T
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)

# Run in the 'shared' queue (job may share node with other jobs)
#SBATCH -p test 
#SBATCH -o outAllPart1Sbatch.out

# Modules necessary for job:
# module purge
# module load gcc openmpi

# gcc -O3 ./serial.c -o serial

# compile part1.c into part1
mpicc -lm -O3 part1.c -o part1

# run part1 with 4 processes (from #SBATCH -n, above)
for Nnum in 32 64 128 256; do
    threads=$((Nnum * Nnum / 1024))
    echo "OMP_NUM_THREADS=$threads Nnum=$Nnum"
    echo "serial"
    time ./serial -NN $Nnum 
    echo "mpi"
    time mpirun -np $threads ./part1 -NN $Nnum
done

