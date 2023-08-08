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
module purge
module load gcc openmpi

# run part1 with 4 processes (from #SBATCH -n, above)
for Nnum in 32 64 128 256; do
    threads=$((Nnum * Nnum / 1024))
    echo "OMP_NUM_THREADS=$threads Nnum=$Nnum"
    echo "serial"
    serial_executable_name="serial$threads"
    time ./"$serial_executable_name"
    echo "mpi"
    MPI_executable_name="part1n$threads"
    time mpirun -np $threads ./"$MPI_executable_name"
done

